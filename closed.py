from __future__ import division
import numpy as np
from data import getData, report
from atom import getOrbitalAtom, getConstant, getValenceElectrons
from math import copysign

def ehtClosed(nAtoms, nBasis, nElectrons, atoms, s, gamma, method):
    #Extended Huckel theory from closed shells

    #retrieve data 
    g = getData('g')
    f = getData('f')
    beta = getData('beta')
    eng = getData('eng')

    nOccupied = nElectrons//2

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

    #density matrix
    density = 2.0 * np.dot(eigenvectors[:, :nOccupied], eigenvectors[:,:nOccupied].T)

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

    return h, density, nOccupied

def scfClosed(nAtoms, nBasis, nOccupied, atoms, h, s, density, gamma, parameters, silent):
    #do an scf computation for closed shell cndo/indo

    #retrieve data 
    g = getData('g')
    f = getData('f')

    #initial Fock matrix is core Hamiltonian
    fock = np.zeros_like(h)
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
            fock[i,i] = h[i,i] - density[i,i] * gamma[k,k] * 0.5 

            for j in range(nBasis):
                l = getOrbitalAtom(j, atoms)
                fock[i,i] += density[j,j] * gamma[k,l]

        for i in range(nBasis-1):
            k = getOrbitalAtom(i, atoms)

            for j in range(i+1, nBasis):
                l = getOrbitalAtom(j, atoms)
                fock[i,j] = fock[j,i] = h[j,i] - density[j,i] * gamma[k,l] * 0.5


        #INDO correction
        if parameters['method'] == 'indo':

            for a in atoms:

                if a.number == 1: continue
                base = a.orbitals[0]
                p_diag = np.sum(np.diag(density, k=0)[base:base+4])

                fock[base, base] += -(p_diag - density[base, base]) * g[a.number-1] / 6.0
                for i in range(1,4):
                    fock[base+i, base+i] += -density[base, base] * g[a.number-1] / 6.0 - \
                                             (p_diag - density[base, base]) * 7.0 * f[a.number-1] / 50.0 + \
                                             density[base+i, base+i] * 11.0 * f[a.number-1] / 50.0

                    fock[base+i, base] += density[base, base+i] * g[a.number-1] / 2.0


                fock[base+2, base+1] += density[base+2,base+1] * 11.0 * f[a.number-1] / 50.0
                fock[base+3, base+1] += density[base+3,base+1] * 11.0 * f[a.number-1] / 50.0
                fock[base+3, base+2] += density[base+3,base+2] * 11.0 * f[a.number-1] / 50.0

                #symmetrize by copying lower diagonal
                fock = np.tril(fock) + np.triu(fock.T,1)

        eSCF = 0.0
        for i in range(nBasis):
            eSCF += 0.5 * density[i,i] * (fock[i,i] + h[i,i])
        for i in range(nBasis-1):
            for j in range(i+1, nBasis):
                eSCF += density[i,j] * (h[i,j] + fock[j,i])

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
        eigenvalues, eigenvectors = np.linalg.eigh(fock)

        #density matrix
        density = 2.0 * np.dot(eigenvectors[:, :nOccupied], eigenvectors[:,:nOccupied].T)
        
        #damping
        if damping:
            if damp_cycle != 1:
                density = alpha * damp_density + (1.0-alpha) * density
            damp_density = density
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
    if parameters['method'] == 'cndo':
        atomic_energy = getData('cndo_atomic_energies')
    else:
        atomic_energy = getData('indo_atomic_energies')


    eBinding = eSCF + eNuclear
    for i in range(nAtoms):
        eBinding -= atomic_energy[atoms[i].number-1]

    if not silent: report('scf_end', matrix=[eSCF, eNuclear, eBinding])

    #output eigensolution
    if not silent: report('eigensolution', matrix=[eigenvalues, eigenvectors, atoms, ''])

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

    if not silent: report('bonds', atoms=atoms, matrix=[density, s, nOccupied])

    return eSCF+eNuclear, eigenvalues