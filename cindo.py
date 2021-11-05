from __future__ import division
import numpy as np
from atom import getConstant, atom, getSlater, assignOrbitals, harmonicTransform, valenceElectrons
from math import factorial
from integral import reducedOverlap
import data

def mol(section, file):
    #read input file

    coordinates = []
    cartesian = True

    #loop over input file
    with open(file,'r') as data:
        f = data.read().split('\n')

        #commands, options and check flags
        valid   = {'name':[False,'string'],'basis':[False,'string'],'charge':[False,'int'],'multiplicity':[False,'int'], 'cycles':[False,'int'],\
                   'tolerance':[False,'float'],'post':[False,'string'],'matrix':[False,'c'],'units':[False,'bohr','angstrom'],\
                   'method':[False,'string'],'shell':[False,'string'],'geometry':[False,'string'],'damping':[False,'string']}
        values  = {'name':'','basis':'slater','charge':0,'multiplicity':1,'cycles':100,'tolerance':1e-8,'post':'','matrix':'c','units':'bohr', \
                   'method':'cndo','shell':'closed','geometry':'','damping':''}

        for nline, line in enumerate(f):

            #find requested molecule
            if section != '':
                if not ('name=' + section) == line:
                    continue
                else:
                    section = ''

            if line.strip() == 'end': break
            
            if (len(line.split()) != 0) and (line.split()[0][0] == '#'): continue

            if '=' in line:

                #string commands
                if any(element in line for element in ['matrix','units']):

                    data = line.split('=')
                    key = data[0]
                    value = data[1]
                    if value in valid[key]:
                        values[key] = value
                        valid[key][0] = True   
                        if value == 'c': cartesian = True

                #integer commands
                if any(element in line for element in ['charge', 'multiplicity','cycles']):

                    data = line.split('=')
                    key = data[0]
                    if data[1].replace('-','',1).isnumeric():
                        values[key] = int(data[1])
                        valid[key][0] = True

                #float command
                if any(element in line for element in ['tolerance']):

                    data = line.split('=')
                    key = data[0]
                    if data[1].replace('e','',1).replace('-','',1).replace('.','').isnumeric():
                        values[key] = float(data[1])
                        valid[key][0] = True

                if any(element in line for element in['name','basis','post','method','shell','geometry','damping']):

                    data = line.split('=')
                    key = data[0]
                    values[key] = data[1]
                    valid[key][0] = True


            else:

                #process atom definitions - 'cartesian' flags cartesian or z-matrix
                data = line.split()
                if (len(data) != 0):
                    if cartesian:
                        coordinates.append([data[0], int(data[1]), float(data[2]), float(data[3]), float(data[4])])             
                if not cartesian:
                    coordinates.append(line)

    #check requested section found
    if not section == '':
        exit('molecule identifier [' + section +'] not found') 

    #check options and defaults [no default for matrix - critical error]
    for i in valid:
        if i[0] == 'False':
            exit('critical input error')

    #check to see if valid basis
    basisName = values['basis']
    if not basisName == 'slater':
        exit('slater only supported')

    #check for valid method
    method = values['method']
    if not method in ['cndo','indo']:
        exit('CNDO and INDO only supported')

    #construct atom object array
    molAtom = []

    #convert to bohr if necessary
    conversion = 1.0
    if values['units'] == 'angstrom':
        conversion = 1.0/getConstant('bohr->angstrom')

    if cartesian:
        for i in range(len(coordinates)):
            molAtom.append(atom(coordinates[i][0], coordinates[i][1], [], [], [coordinates[i][2]*conversion, \
                                                                       coordinates[i][3]*conversion, \
                                                                       coordinates[i][4]*conversion]))
    else:
        z = zMatrix(coordinates)
        for i in range(len(z)):
            molAtom.append(atom(z[i][0], int(z[i][1]), [], [], [float(z[i][2])*conversion, \
                                                        float(z[i][3])*conversion, \
                                                        float(z[i][4])*conversion]))

    #add Slater exponents and orbital range to atom objects
    molAtom = getSlater(molAtom)
    molAtom = assignOrbitals(molAtom)

    #return atom object array and [name,basis,charge,multiplicity,cycles,tolerance,units,method,shell]

    return molAtom, values

def scf(section='', file='project.hpf', silent=False, method = 'cndo'):
    #main scf routine

    molAtom, parameters = mol(section, file)
    nAtoms = len(molAtom)

    #is the method override active
    parameters['method'] = method

    #angular and magnetic quantum numbers - s,px,py,pz,dzz,dxz,dyz,dx-y,dxy
    orbitals = [[0,0],[1,1],[1,-1],[1,0],[2,0],[2,1],[2,-1],[2,2],[2,-2]]

    #get number of basis functions
    nbf = molAtom[-1].orbitals[1] + 1

    #write input info
    if not silent: data.report('header', molAtom, parameters)

    s = np.zeros((nbf, nbf))
    gamma = np.zeros((nAtoms, nAtoms))

    #step through the pairs of atoms
    for i in range(nAtoms):
        for j in range(i, nAtoms):
            
            ia = molAtom[i] ; ja = molAtom[j]

            #get unit vector along atoms axis
            e = ja.center - ia.center
            r = np.linalg.norm(e)
            if r > 1e-6: e /= r

            #loop over orbitals in each atom, get number of orbitals on pair
            nbf_a = ia.orbitals[1]-ia.orbitals[0]+1
            nbf_b = ja.orbitals[1]-ja.orbitals[0]+1
            pair = np.zeros((nbf_a, nbf_b))

            for ib in range(nbf_a):
                for jb in range(nbf_b):

                    if ia.id == ja.id:
                        if ib == jb: pair[ib, jb] = 1.0      #same atom, same orbital
                        else: pair[ib, jb] = 0.0             #same atom, different orbital
                    else:
                        #get magnetic quantum number
                        mi = orbitals[ib][1]                 
                        mj = orbitals[jb][1]

                        if mi != mj:                         #different atom, different m
                            pair[ib, jb] = 0.0
                        elif mi < 0:                         #different atom, py,dyz,dxy
                            pair[ib, jb] = pair[ib-1, jb-1]
                        else:                                #different atom, same m
                            pair[ib, jb] = np.sqrt(pow(ia.slater[1]*r,(2.0*ia.slater[0]+1))*pow(ja.slater[1]*r,(2.0*ja.slater[0]+1)) / \
                                           (factorial(2.0*ia.slater[0])*factorial(2.0*ja.slater[0]))) * pow(-1, orbitals[jb][0]+orbitals[jb][1]) * \
                                           reducedOverlap(ia.slater[0], orbitals[ib][0], orbitals[ib][1], ja.slater[0], orbitals[jb][0], ia.slater[1]*r, ja.slater[1]*r)

            #rotate back to molecular basis
            if r > 1e-6:
                t = harmonicTransform(max(orbitals[nbf_a-1][0], orbitals[nbf_b-1][0]), e)

                temp = np.dot(pair, t.T[:nbf_b, :nbf_b])
                pair = np.dot(t[:nbf_a, :nbf_a], temp)

            #enter overlap values
            for im in range(nbf_a):
                lower_a = ia.orbitals[0] + im
                for jm in range(nbf_b):
                    lower_b = ja.orbitals[0] + jm
                    s[lower_a, lower_b] = pair[im,jm]

            t = [0.0, 0.0]
            #begin Coulomb 1-center integrals
            if ia.id == ja.id:
                t[0] = factorial(2*ia.slater[0]-1)/pow(2.0*ja.slater[1],2*ia.slater[0])
                for k in range(1, 2*ia.slater[0]+1):
                    t[1] += (k*pow(2*ia.slater[1],2*ia.slater[0]-k)*factorial(4*ia.slater[0]-k-1))/ \
                            (factorial(2*ia.slater[0]-k)*2*ia.slater[0]*pow(2*(ia.slater[1]+ja.slater[1]),4*ia.slater[0]-k))
            else:
            #begin Coulomb 2-center integrals
                t[0] = pow(0.5*r,2*ja.slater[0])*reducedOverlap(0, 0, 0, 2*ja.slater[0]-1, 0, 0.0, 2*ja.slater[1]*r)
                for k in range(1, 2*ia.slater[0]+1):
                    t[1] += ((k*pow(2*ia.slater[1],2*ia.slater[0]-k)*pow(0.5*r, 2*ia.slater[0]-k+2*ja.slater[0]))/ \
                            (factorial(2*ia.slater[0]-k)*2*ia.slater[0])) * \
                            reducedOverlap(2*ia.slater[0]-k, 0, 0, 2*ja.slater[0]-1, 0, 2*ia.slater[1]*r, 2*ja.slater[1]*r)  

            gamma[i,j] = (pow(2*ja.slater[1],2*ja.slater[0]+1)/factorial(2*ja.slater[0]))*(t[0]-t[1])

    #symmetrize s-matrix
    s += s.T - np.diag(s.diagonal())
    if not silent: data.report('overlap', matrix=s)

    #symmetrize gamma-matrix
    gamma += gamma.T - np.diag(gamma.diagonal())
    if not silent: data.report('coulomb',matrix=gamma)

    if parameters['shell'] == 'closed':
        from closed import ehtClosed, scfClosed

        h, density, nOccupied = ehtClosed(nAtoms, nbf, valenceElectrons(molAtom, parameters['charge']), molAtom, s, gamma, parameters['method'])
        if not silent: data.report('core Hamiltonian',matrix=h)

        eSCF, eigenvalues = scfClosed(nAtoms, nbf, nOccupied, molAtom, h, s, density, gamma, parameters, silent)

        return eSCF, eigenvalues

    elif parameters['shell'] == 'open':
        from open import ehtOpen, scfOpen

        h, alpha_density, beta_density, alpha_electrons, beta_electrons = ehtOpen(nAtoms, nbf, valenceElectrons(molAtom, parameters['charge']), \
                                                                             parameters['multiplicity'], molAtom, s, gamma, parameters['method'])
        if not silent: data.report('core Hamiltonian',matrix=h)
         
        eSCF, eigenvalues = scfOpen(nAtoms, nbf, alpha_electrons, beta_electrons, molAtom, h, s, alpha_density, beta_density, gamma, parameters, silent)

        return eSCF, eigenvalues[0], eigenvalues[1]