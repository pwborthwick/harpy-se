from __future__ import division
import numpy as np

symbol = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']

covalentRadius = [31, 28, 128, 96, 84, 73, 71, 66, 57, 58, 166, 141, 121, 111, 107, 105, 102, 106] 

class atom(object):

    def __init__(self,id,number, slater=[], orbitals=[], center= np.zeros(3)):      
        self.id = id
        self.number = number
        self.slater = slater
        self.orbitals = orbitals
        self.center = np.array(center)

def getSymbol(n):
    #get atom symbol from z

    return symbol[n-1]

def getConstant(unit, POPLE=True):
    nist = {'bohr->angstrom' : 0.52917721092,  'hartree->eV' : 27.21138505, 'au->debye' : 2.541580253, \
            'hartree->kcal' : 627.509474, 'hartree->kJ' : 2625.499640, 'picometre->bohr' : 0.018897261339213}
    pople = {'bohr->angstrom' : 0.529167, 'hartree->eV' : 27.21, 'hartree->kcal' : 627.509474, 'hartree->kJ' : 2625.499640, \
             'au->debye' : 2.5416, 'picometre->bohr' : 0.018897261339213}

    if POPLE:
        return pople[unit]
    else : 
        return nist[unit]

def getSlater(atoms):
    #assign the slater exponents - [n, mu]

    slater = [[1, 1.2], [1, 1.7]]
    for a in range(3, len(symbol)+1):

        if a in range(3, 11): 
            slater.append([2, 0.325 * (a-1)])
        else:
            slater.append([3, (0.65 * (a) - 4.95)/3.0])

    for a in atoms:
        a.slater = slater[a.number-1]

    return atoms

def assignOrbitals(atoms):
    #attach orbital range for each atom

    nbf = [1,1,4,4,4,4,4,4,4,4,9,9,9,9,9,9,9,9]

    n = 0
    for a in atoms:

        a.orbitals = [n, n + nbf[a.number-1]-1]
        n += nbf[a.number-1]

    return atoms

def valenceElectrons(atoms, charge):
    #compute number of non-core electrons

    n = 0
    for a in atoms:
        n += getValenceElectrons(a.number)

    return n - charge

def getOrbitalAtom(orbital, atoms):
    #get the atom a particular orbital is attached to

    for i, a in enumerate(atoms):
        span = range(a.orbitals[0], a.orbitals[1]+1)
        if orbital in span: return i

    return -1

def getValenceElectrons(atom):
    #get the valence electron count for atom

    valence = [1,2,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]

    return valence[atom-1]

def harmonicTransform(max_l, e):
    #rotate from diatomic basis to molecular basis

    from math import sqrt
    t = np.zeros((9,9))
    
    #deal with small angle
    ct = e[2]
    st = sqrt(1.0 - ct*ct)
    if (st < 1e-6): 
        st = 0.0
        cp = 1.0
        sp = 0.0
    else:
        cp = e[0]/st
        sp = e[1]/st
        
    t[0,0] = 1.0
    #d-orbital in pair
    if max_l > 1:
        c2t = ct*ct - st*st
        s2t = 2.0*st*ct
        c2p = cp*cp - sp*sp
        s2p = 2.0*sp*cp

        t[4,4] = 0.5*(3.0*ct*ct-1.0)   ; t[4,5] = -0.5*sqrt(3)*s2t ; t[4,7] = 0.5*sqrt(3)*st*st
        t[5,4] = 0.5*sqrt(3)*s2t*cp    ; t[5,5] = c2t*cp           ; t[5,6] = -ct*sp            ; t[5,7] = t[5,4]/sqrt(3)      ; t[5,8] = st*sp
        t[6,4] = 0.5*sqrt(3)*s2t*sp    ; t[6,5] = c2t*sp           ; t[6,6] = ct*cp             ; t[6,7] = -t[6,4]/sqrt(3)     ; t[6,8] = -st*cp
        t[7,4] = 0.5*sqrt(3)*st*st*c2p ; t[7,5] = 0.5*s2t*c2p      ; t[7,6] = -st*s2p           ; t[7,7] = 0.5*(1.0+ct*ct)*c2p ; t[7,8] = -ct*s2p
        t[8,4] = 0.5*sqrt(3)*st*st*s2p ; t[8,5] = 0.5*s2t*s2p      ; t[8,6] = st*c2p            ; t[8,7] = 0.5*(1.0+ct*ct)*s2p ; t[8,8] = ct*c2p

    #p-orbital in pair
    if max_l > 0:
        t[1,1] = ct*cp ; t[1,2] = -sp ; t[1,3] = st*cp
        t[2,1] = ct*sp ; t[2,2] = cp  ; t[2,3] = st*sp
        t[3,1] = -st   ; t[3,3] = ct

    return t

def getGeometry(geometry, atoms):
    #evaluate the requested geometry attributes

    metrics = []
    while True:
        i = geometry.find('{')
        if i == -1: break
        j = geometry.find('}')

        instruction = geometry[i+1:j]
        geometry =geometry[j+1:]

        operation = instruction[0]
        instruction_string = instruction[2:]
        instruction = instruction_string.split(',')

        if operation == 'r':
            metrics.append(['bond', instruction_string, np.linalg.norm(atoms[int(instruction[0])-1].center - atoms[int(instruction[1])-1].center)* \
                            getConstant('bohr->angstrom')] )
        elif operation == 'a':
            from math import acos

            i = atoms[int(instruction[1])-1].center - atoms[int(instruction[0])-1].center
            j = atoms[int(instruction[2])-1].center - atoms[int(instruction[1])-1].center
            metrics.append(['angle', instruction_string, 180*acos(np.dot(i,j)/(np.linalg.norm(i)*np.linalg.norm(j)))/np.pi])

    return metrics

def isBond(atoms,i,j):
    #if sum of covalent radii of i and j is less than seperation

    if i == j: return False
    sumRadii = (covalentRadius[atoms[i].number-1] + covalentRadius[atoms[j].number-1]) * getConstant('picometre->bohr')

    if np.linalg.norm(atoms[i].center-atoms[j].center) < 1.6 * sumRadii:
        return True
    else:
        return False


