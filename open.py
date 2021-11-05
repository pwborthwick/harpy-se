from __future__ import division
import numpy as np
from data import getData, report
from atom import getOrbitalAtom, getConstant, getValenceElectrons
from math import copysign

def ehtOpen(nAtoms, nBasis, nElectrons, multiplicity, atoms, s, gamma, method):
    #Extended Huckel theory from closed shells

    #retrieve data 
    g = getData('g')
    f = getData('f')
    beta = getData('beta')
    eng = getData('eng')

    #build Hamiltonian
    h = s.copy()
    #get diagonal Hamiltonian elements
    for a in atoms:
        k = 0
        for i in range(a.orbitals[0], a.orbitals[1]+1):
            k += 1
            if k == 1: h[i,i] = -eng[a.number-1, 0]
            elif k < 5: h[i,i] = -eng[a.number-1,1]
            else: h[i,i] = -eng[a.number-1,2]

            h[i,i] /= getConstant('hartree->eV')

    #spin occupancy
    alpha_electrons = (nElectrons + multiplicity - 1)//2
    beta_electrons  = (nElectrons - multiplicity + 1)//2

    #get off-diagonal Hamiltonian elements
    for i in range(1, nBasis):
        
        for j in range(i):

            zi = atoms[getOrbitalAtom(i, atoms)].number ; zj = atoms[getOrbitalAtom(j, atoms)].number
            if (zi > 9) or (zj > 9):
                h[i,j] = 0.75 * h[i,j]*(beta[zi-1] + beta[zj-1])
            else:
                h[i,j] = h[i,j]*(beta[zi-1] + beta[zj-1])

            #convert to hartree and symmetrize
            h[i,j] /= getConstant('hartree->eV')*2.0
            h[j,i] = h[i,j]

    #diagonalise Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(h)

    #density matrices
    alpha_density = np.zeros_like(h)
    beta_density = np.zeros_like(h)
    alpha_density = np.dot(eigenvectors[:, :alpha_electrons], eigenvectors[:,:alpha_electrons].T)    
    beta_density =  np.dot(eigenvectors[:, :beta_electrons],  eigenvectors[:,:beta_electrons].T)

    #add coulomb to core Hamiltonian
    q = np.einsum('ii->i', h)

    for i in range(nBasis):

        k = getOrbitalAtom(i, atoms)
        q[i] += 0.5 * gamma[k,k]

        for j in range(nAtoms):
            q[i] -= getValenceElectrons(atoms[j].number) * gamma[k,j]

    #INDO corrections
    if method == 'indo':
        for a in atoms:

            if a.number == 1: continue
            elif a.number in [2,5,6,7,8,9]:
                t = g[a.number-1]/3.0 + (getValenceElectrons(a.number) - 2.5) * 2.0 * f[a.number-1] / 25.0
            elif a.number == 3:
                t = g[a.number-1] / 12.0
            elif a.number == 4:
                t = g[a.number-1] / 4.0

            if a.number in range(4,10):
                q[a.orbitals[0]] += (getValenceElectrons(a.number) - 1.5) * g[a.number-1] / 6.0

            q[a.orbitals[0]+1:a.orbitals[0]+4] += t

    for i in range(nBasis):
        h[i,i] = q[i]

    return h, alpha_density, beta_density, alpha_electrons, beta_electrons

def scfOpen(nAtoms, nBasis, alpha_electrons, beta_electrons, atoms, h, s, alpha_density, beta_density, \
            gamma, parameters, silent):
    #do an scf computation for closed shell cndo/indo

    #retrieve data 
    g = getData('g')
    f = getData('f')

    #initial Fock matrix is core Hamiltonian
    alpha_fock = np.zeros_like(h)
    beta_fock  = np.zeros_like(h)
    preEnergy = 0.0

    #store record of convergence behaviour
    scf_trend = ''

    #damping control
    damp_command = parameters['damping']
    damping = (damp_command != '')
    if damping:
        alpha, damp_iterations = damp_command.split(',')
        alpha = float(alpha.replace('{','')) ; damp_iterations = int(damp_iterations.replace('}',''))
        damp_cycle = 1

    for cycle in range(parameters['cycles']):

        #fock matrix construction
        for i in range(nBasis):
            k = getOrbitalAtom(i, atoms)
            alpha_fock[i,i] = h[i,i] - alpha_density[i,i] * gamma[k,k]
            beta_fock[i,i]  = h[i,i] -  beta_density[i,i] * gamma[k,k]

            for j in range(nBasis):
                l = getOrbitalAtom(j, atoms)
                alpha_fock[i,i] += (alpha_density[j,j]+beta_density[j,j]) * gamma[k,l] 
                beta_fock[i,i]  += (alpha_density[j,j]+beta_density[j,j]) * gamma[k,l]

        for i in range(nBasis-1):
            k = getOrbitalAtom(i, atoms)

            for j in range(i+1, nBasis):
                l = getOrbitalAtom(j, atoms)
                alpha_fock[i,j] = alpha_fock[j,i] = h[j,i] - alpha_density[j,i] * gamma[k,l]
                beta_fock[i,j]  = beta_fock[j,i]  = h[j,i] -  beta_density[j,i] * gamma[k,l]


        #INDO correction
        if parameters['method'] == 'indo':

            for a in atoms:

                if a.number == 1: continue
                base = a.orbitals[0]
                p_diag_alpha = np.sum(np.diag(alpha_density, k=0)[base:base+4])
                p_diag_beta = np.sum(np.diag(beta_density, k=0)[base:base+4])

                alpha_fock[base, base] += -(p_diag_alpha - alpha_density[base, base]) * g[a.number-1] / 3.0
                beta_fock[base, base]  += -(p_diag_beta - beta_density[base, base]) * g[a.number-1] / 3.0

                for i in range(1,4):
                    alpha_fock[base+i, base+i] += (alpha_density[base+i, base+i] - (p_diag_alpha-alpha_density[base,base])) * \
                                                  f[a.number-1]/5.0 - alpha_density[base,base] * g[a.number-1]/3.0  + \
                                                  (6.0 * beta_density[base+i,base+i] - 2.0 * (p_diag_beta-beta_density[base,base])) * \
                                                  f[a.number-1]/25.0
                    beta_fock[base+i, base+i]  += (beta_density[base+i, base+i] - (p_diag_beta-beta_density[base,base])) * \
                                                  f[a.number-1]/5.0 - beta_density[base,base] * g[a.number-1]/3.0  + \
                                                  (6.0 * alpha_density[base+i,base+i] - 2.0 * (p_diag_alpha-alpha_density[base,base])) * \
                                                  f[a.number-1]/25.0
                    alpha_fock[base+i, base] += (alpha_density[base+i,base] + 2.0*beta_density[base+i, base]) * g[a.number-1] / 3.0
                    beta_fock[base+i, base]  += (beta_density[base+i,base] + 2.0*alpha_density[base+i, base]) * g[a.number-1] / 3.0

                    for j in range(1,4):
                        if i == j: continue
                        if j > i:
                            alpha_fock[base+j,base+i] += (5.0 * alpha_density[base+j,base+i] + 6.0 * beta_density[base+j,base+i]) * f[a.number-1] / 25.0
                        else:
                            beta_fock[base+i,base+j]  += (5.0 * beta_density[base+j,base+i] + 6.0 * alpha_density[base+j,base+i]) * f[a.number-1] / 25.0

            #symmetrize by copying lower diagonal
            alpha_fock = np.tril(alpha_fock) + np.triu(alpha_fock.T,1)
            beta_fock =  np.tril(beta_fock) + np.triu(beta_fock.T,1)

        eSCF = 0.0
        for i in range(nBasis):
            eSCF += 0.5 * (alpha_density[i,i] * (alpha_fock[i,i] + h[i,i]) + \
                            beta_density[i,i] * (beta_fock[i,i]  + h[i,i])) 
        for i in range(nBasis-1):
            for j in range(i+1, nBasis):
                eSCF += alpha_density[i,j] * (h[i,j] + alpha_fock[j,i]) + beta_density[i,j] * (h[i,j] + beta_fock[j,i])

        if not silent: report('scf_energy', matrix = [cycle,eSCF,preEnergy,scf_trend])

        #get scf convergence trend and test for oscillation
        scf_trend += str(copysign(1,(eSCF - preEnergy)))[0].replace('1','+')
        if '-+-+-' in scf_trend:
            exit('oscillatory scf - exit')

        #convergence check
        if abs(preEnergy - eSCF) < parameters['tolerance']:
            break

        #not converged - prepare next iteration
        preEnergy = eSCF
        eigenvalues, eigenvectors = np.linalg.eigh(alpha_fock)
        eigensolution = [eigenvalues, eigenvectors]

        #density matrix alpha
        alpha_density = np.dot(eigenvectors[:, :alpha_electrons], eigenvectors[:,:alpha_electrons].T)

        eigenvalues, eigenvectors = np.linalg.eigh(beta_fock)
        eigensolution.append(eigenvalues) ; eigensolution.append(eigenvectors)

        #density matrix beta
        beta_density = np.dot(eigenvectors[:, :beta_electrons], eigenvectors[:,:beta_electrons].T)
        
        #damping
        if damping:
            if damp_cycle != 1:
                alpha_density = alpha * alpha_damp_density + (1.0-alpha) * alpha_density
                beta_density = alpha  * beta_damp_density  + (1.0-alpha) * beta_density
            alpha_damp_density = alpha_density ; beta_damp_density = beta_density
            damp_cycle += 1
            damping = (damp_cycle != damp_iterations)

    if cycle == (parameters['cycles']-1):
        if not silent: report('scf_end', matrix=[0.0000])
        exit('not converged')

    eNuclear = 0.0
    for i in range(nAtoms-1):
        for j in range(i+1, nAtoms):
            r = np.linalg.norm(atoms[j].center - atoms[i].center)
            eNuclear += getValenceElectrons(atoms[i].number) * getValenceElectrons(atoms[j].number)/r

    #get binding energy using atomic energies
    atomic_energy = getData('cndo_atomic_energies')

    eBinding = eSCF + eNuclear
    for i in range(nAtoms):
        eBinding -= atomic_energy[atoms[i].number-1]

    if not silent: report('scf_end', matrix=[eSCF, eNuclear, eBinding])

    #output eigensolution
    if not silent: 
        report('eigensolution', matrix=[eigensolution[0], eigensolution[1], atoms, '\u03B1'])
        report('eigensolution', matrix=[eigensolution[2], eigensolution[3], atoms, '\u03B2'])

    #for charges and dipole combine densities
    density = alpha_density + beta_density

    #population analysis and dipole
    charges = [0.0]*nAtoms
    for i,a in enumerate(atoms):
        for j in range(a.orbitals[0], a.orbitals[1]+1):
            charges[i] += density[j,j]

    #dipole calculation
    dipole = np.zeros((3)); mu_density = np.zeros((3)); mu_sp = np.zeros((3)); mu_pd = np.zeros((3))
    empirical = [0.65, 4.95, 10.27175, 7.33697, 0.325]

    for i, a in enumerate(atoms):

        if (a.number < 11) and (a.number >= 3):
            mu_sp -= density[a.orbitals[0], a.orbitals[0]+1:a.orbitals[1]+1] * empirical[3]/(empirical[4] * (a.number-1))
        elif (a.number >= 11):
            slater = (empirical[0] * a.number - empirical[1])/3.0
            factor = (getConstant('au->debye')*7.0)  / (slater * np.sqrt(5))
            k = a.orbitals[0]
            mu_sp -= density[k, k+1:k+4] * empirical[2] / slater
            mu_pd += [density[k+1,k+4]*factor*(density[k+1,k+7]+density[k+2,k+8]+density[k+3,k+5]-(1.0/np.sqrt(3))), \
                      density[k+2,k+4]*factor*(density[k+1,k+8]+density[k+2,k+7]+density[k+3,k+6]-(1.0/np.sqrt(3))), \
                      density[k+3,k+4]*factor*(density[k+1,k+5]+density[k+2,k+6]+(2.0/np.sqrt(3)))]

        mu_density += (getValenceElectrons(a.number)-charges[i]) * a.center * getConstant('au->debye')

    if not silent: report('population', matrix=[charges, atoms, density, mu_density, mu_sp, mu_pd], parameters=parameters)

    if not silent: report('spin', atoms=atoms, matrix=[eigensolution, s, [alpha_electrons, beta_electrons], [alpha_density, beta_density], \
                                                       ['\u03B1','\u03B2']], parameters=parameters)

    return eSCF + eNuclear, [eigensolution[0], eigensolution[2]]
