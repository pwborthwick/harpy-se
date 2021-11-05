from __future__ import division
import numpy as np
from atom import getSymbol, getConstant, getValenceElectrons, getGeometry, valenceElectrons

y = { -288.0 : [6899] ,\
      -192.0 : [6896, 6907, 6891] ,\
      -160.0 : [7430] ,\
      -144.0 : [4375, 4397, 1909, 1893, 1922, 1906] ,\
      -128.0 : [7032, 7033, 5322, 8014, 7076, 7079, 7572, 7438] ,\
      -96.0  : [6904, 6915, 2720, 2715, 5185, 5186, 5206, 5198, 4370, 4381, 8020, 8031, 7435, 7436, 7446, 7447, 5545, 5557] ,\
      -64.0  : [7049, 7041, 7035, 7027, 5340, 5326, 5180, 5190, 8156, 8165, 8148, 8150, 7084, 7071, 7580, 7566, 5681, 5673, 5691, 5685, 8200, 8194, 7615, 7616, 7618, 7610, 5725, 5729] ,\
      -48.0  : [2719, 2703, 2712, 2704, 2731, 2714, 2723, 2707, 4386, 3079, 3071, 3063, 3082, 3074, 3066] ,\
      -32.0  : [6892, 5188, 5189, 5181, 5182, 8015, 7431, 5539, 5540, 5549, 5541] ,\
      -16.0  : [2854, 2856, 2865, 2840, 2849, 2851, 4393, 3205, 3215, 3216, 3199, 3200, 3210, 3259, 3243, 3254] ,\
      -6.0   : [955] ,\
       1.0   : [952] ,\
       2.0   : [975] ,\
       3.0   : [964] ,\
       6.0   : [948, 970] ,\
       8.0   : [959] ,\
       16.0  : [2863, 2842, 4372, 973, 977, 3214, 3206, 3198, 3217, 3209, 3201, 3250, 3261, 3245] ,\
       32.0  : [6913, 2847, 2858, 5194, 5204, 5187, 5197, 950, 961, 8029, 7444, 5554, 5546, 5555, 5556] ,\
       48.0  : [2710, 2711, 2729, 2713, 2722, 2706, 2724, 2716, 4379, 966, 3070, 3080, 3081, 3064, 3065, 3075] ,\
       64.0  : [7039, 7040, 7025, 7026, 5329, 5315, 5195, 5196, 8155, 8157, 8149, 8158, 7085, 7070, 7579, 7565, 5680, 5674, 5692, 5684, 8201, 8193, 7625, 7608, 7617, 7609, 5718, 5736] ,\
       96.0  : [6890, 6901, 2721, 2705, 5178, 5179, 5199, 5191, 4384, 4395, 8013, 8024, 7428, 7429, 7439, 7440, 5538, 5550] ,\
       128.0 : [7042, 7034, 5333, 8021, 7086, 7069, 7573, 7437] ,\
       144.0 : [4368, 4390, 1900, 1920, 1895, 1915] ,\
       160.0 : [7445] ,\
       192.0 : [6905, 6889, 6900] ,\
       288.0 : [6906] }

z = { -10.0 : [666, 158] , \
      -6.0  : [226, 310] ,\
      -5.0  : [668, 162, 412, 530] ,\
      -4.0  : [225, 229, 533] ,\
      -3.0  : [345, 415, 546, 549, 700, 326, 329] ,\
      -2.0  : [313, 314, 565, 446, 462] ,\
      -1.0  : [341, 664, 154, 222, 230, 307, 409, 411, 528, 562, 732, 580, 581, 596, 443, 447, 698, 325, 330] ,\
       1.0  : [347, 669, 164, 223, 231, 315, 414, 416, 534, 566, 733, 545, 550, 579, 582, 598, 444, 448, 701, 324, 331, 460, 464] ,\
       2.0  : [308, 309, 563, 547, 548, 445] ,\
       3.0  : [343, 410, 699, 327, 328] ,\
       4.0  : [224, 228, 529] ,\
       5.0  : [665, 156, 413, 532] ,\
       6.0  : [227, 312] ,\
       10.0 : [667, 160] }

g = [0.0, 0.0, 0.092012, 0.1407, 0.199265, 0.267708, 0.346029, 0.43423, 0.532305]
f = [0.0, 0.0, 0.049865, 0.089125, 0.13041, 0.17372, 0.219055, 0.266415, 0.31580]
beta = [-9.0, 0.0, -9.0, -13.0, -17.0, -21.0, -25.0, -31.0, -39.0, 0.0, -7.7203, -9.4471, -11.3011, -13.065, -15.070, -18.150, -22.330, 0.0]

eng =   [7.1761,0.0000,0.0000,0.0000,0.0000,0.0000,3.1055,1.2580,0.0000,5.94557,2.5630,0.0000,9.59407,4.0010,0.0000, \
        14.0510,5.5720,0.0000,19.31637,7.2750,0.0000,25.39017,9.1110,0.0000,32.2724,11.0800,0.0000,0.0000,0.0000,0.0000, \
        2.804,1.302,0.1500,5.1254,2.0516,0.1619,7.7706,2.9951,0.2242,10.032,4.1325,0.3370,14.032,5.4638,0.5000 ,6.9890, \
        17.649,0.7132,21.590,8.7081,0.9769,0.0000,0.0000,0.0000]

cndo_atomic_energies = [-0.6387302462, 0.00000000000, -0.2321972405, -1.1454120355, -2.977423904, -6.1649936261, -11.0768746252, \
                        -18.0819658651,-27.5491302880, 0.00000000000, -0.1977009568, -0.8671913833, -2.0364557744,-3.8979034686,  \
                        -6.7966009163, -10.7658174340,-16.046701794, 0.00000000000]

indo_atomic_energies = [-0.6387302462, 0.00000000000, -0.2321972405, -1.1219620354, -2.8725750048, -5.9349548261, -10.6731741251, \
                       -17.2920850650, -26.2574377875]

ciso = [539.8635, 0.0, 0.0, 0.0, 0.0, 820.0959, 379.3557, 888.6855, 44829.2]

def getData(type):
      #reshape and present data in correct form

    if type == 'y':
        dim_y_1d = 9135
        dim_y_2d = (9, 5, 203)

        array = np.zeros((dim_y_1d))
        for key in y.keys():
            for i in y[key]:
                array[i-1] = key
        
        array = array.reshape(dim_y_2d, order='F')

        return array 

    elif type == 'z':
        dim_z_1d = 765
        dim_z_2d = (17, 45)

        array = np.zeros((dim_z_1d))
        for key in z.keys():
            for i in z[key]:
                array[i-1] = key

        array = array.reshape(dim_z_2d, order='F')

        return array

    elif type in ['g','f','beta', 'cndo_atomic_energies','indo_atomic_energies','ciso']:
        if type == 'f': return np.array(f)
        if type == 'g': return np.array(g)
        if type == 'beta': return np.array(beta)
        if type == 'cndo_atomic_energies': return np.array(cndo_atomic_energies)
        if type == 'indo_atomic_energies': return np.array(indo_atomic_energies)
        if type == 'ciso': return np.array(ciso)

    elif type == 'eng':

        return np.array(eng).reshape(18,3, order='C')

def report(mode, atoms=None, parameters='',matrix=np.zeros((1,1))):
         #write output

    quantum_numbers = [['0,0,0','s'],['1,1,1','px'],['1,1,-1','py'],['1,1,0','pz'],['2,2,0','dzz'], \
                        ['2,2,1','dxz'],['2,2,-1','dyz'],['2,2,2','dx-y'],['2,2,-2','dxy']]

    np.set_printoptions(precision=4,suppress=True,linewidth=160)
    if mode == 'header':
        print("{:<20} {:<20}".format('method', parameters['method']))
        print("{:<20} {:<20}".format('ident',  parameters['name']))
        print("{:<20} {:<20}".format('basis',  parameters['basis']))
        print("\n{:<20} {:<20}".format('charge',  parameters['charge']))
        print("{:<20} {:<20}".format('multiplicity',  parameters['multiplicity']))
        print("{:<20} {:<20}".format('shell type',  parameters['shell']))
        print("\n{:<20} {:<20}".format('unit',  parameters['units']))
        print("{:<20} {:<20}".format('number of atoms', len(atoms)))
        print('\n    type             coordinates           orbitals   slater')
        for n, i in enumerate(atoms):
            print("{:>2d}   {:<2} {:>10.4f} {:>10.4f} {:>10.4f}     {:>2d}-{:<2d}  {:>2d}:{:>5.3f}".\
                  format(n+1, getSymbol(i.number), i.center[0], i.center[1], i.center[2], \
                                        i.orbitals[0], i.orbitals[1], i.slater[0], i.slater[1]))

        #user requested geometry
        if parameters['geometry'] != '':
            print('\n requested geometry')
            geometry = parameters['geometry']
            metrics = getGeometry(geometry, atoms)
            for m in metrics:
                print(' {:<9}   {:<12}  {:<6.2f}'.format(m[0], m[1], m[2]))

        #occupation numbers
        if parameters['shell'] == 'closed':
            occupied = valenceElectrons(atoms, parameters['charge']) // 2
            occupation = [2] * occupied
        else:
            valence  = valenceElectrons(atoms, parameters['charge'])
            unpaired = parameters['multiplicity'] - 1
            paired   = (valence - unpaired)//2
            occupation = [2] * paired + [1] * unpaired
            occupied = len(occupation)

        #basis orbitals
        print('\n orbitals \n    atom   n,l,m    type')
        n = 1
        for i in atoms:
            m = 0
            for j in range(i.orbitals[1]-i.orbitals[0]+1):

                if n <= occupied: state = 'occupied(' + str(occupation[n-1]) + ')'
                else: state = ''
                print(" {:>2d}   {:<4}  {:<6}    {:<4}   {:>8}".format(n, i.id, quantum_numbers[m][0], \
                                                                      quantum_numbers[m][1], state))
                m += 1
                n += 1

    #matrix outout
    if mode in ['overlap','coulomb','core Hamiltonian','spin density']:
        print('\n', mode, 'matrix')
        print(matrix)

    #self-consistent energy process
    if mode == 'scf_energy':
        if matrix[0] == 0: 
            print('\n SCF iterations\n cycle          energy (au)        delta     trend')
            print('   {:>3d}         {:>10.6f}'.format(matrix[0]+1, matrix[1]))
        else:
            print('   {:>3d}         {:>10.6f}    {:>12.6f}    {:<}'.format(matrix[0]+1, matrix[1], matrix[2]-matrix[1], matrix[3]))

    #final energies
    if mode == 'scf_end':
        if matrix[0] == 0.0000:
            print('\n SCF iterations did not converge')
        else:
            print('\n SCF iterations converged\n')
            print(' electronic energy   {:>10.6f} au'.format(matrix[0])) 
            print(' nuclear energy      {:>10.6f} au'.format(matrix[1]))
            eTotal = matrix[0]+matrix[1]
            print(' total energy        {:>10.6f} au   {:>10.4f} eV   {:>10.4f} kcal/mol   {:>10.4f} kJ/mol'. \
                  format(eTotal, eTotal*getConstant('hartree->eV'), eTotal*getConstant('hartree->kcal'), eTotal*getConstant('hartree->kJ')))
            print('\n binding energy      {:>10.6f} au   {:>10.4f} eV'.format(matrix[2], matrix[2]*getConstant('hartree->eV') ))

    #eigenvalues and eigenvectors
    if mode == 'eigensolution':
        if not matrix[3] == '':
            print('\n eigensolution ' + matrix[3] + '-spin')
        else:
            print('\n eigensolution')

        print('',matrix[0],'\n')
        orbitals = '   '
        for a in matrix[2]:
            for i in range(a.orbitals[1]-a.orbitals[0]+1):
                orbitals += (getSymbol(a.number) + ' ' + quantum_numbers[i][1]).center(8)
        print(orbitals)
        print(matrix[1])

    #charges and dipoles
    if mode == 'population':
        print('\n density matrix\n')
        print(matrix[2])

        if parameters['shell'] == 'closed':
            print('\n charges\n atom    density     charge')
            for i, a in enumerate(matrix[1]):
                print('  {:2}      {:>5.2f}      {:>5.2f} '.format(getSymbol(a.number), matrix[0][i], getValenceElectrons(a.number) - matrix[0][i]))

        #only print dipole if no charge
        if parameters['charge'] == 0: 

            print('\n dipole (debye)\n','                 x         y         z')
            print(' {:<12}{:>8.4f}  {:>8.4f}  {:>8.4f}'.format('densities', matrix[3][0], matrix[3][1], matrix[3][2]))
            print(' {:<12}{:>8.4f}  {:>8.4f}  {:>8.4f}'.format('sp', matrix[4][0], matrix[4][1], matrix[4][2]))
            print(' {:<12}{:>8.4f}  {:>8.4f}  {:>8.4f}'.format('pd', matrix[5][0], matrix[5][1], matrix[5][2]))
            mu_total = matrix[3] + matrix[4] + matrix[5]
            print(' {:<12}{:>8.4f}  {:>8.4f}  {:>8.4f}'.format('total', mu_total[0], mu_total[1], mu_total[2]))

    #bonds
    if mode == 'bonds':
        from atom import isBond
        print('\n inferred bonds by covalent radii')
        for i, a in enumerate(atoms):
            for j, b in enumerate(atoms[i+1:]):
                if isBond(atoms,i, j+i+1): 

                    bond_order = 0.0
                    for p in range(a.orbitals[0], a.orbitals[1]+1):
                        for q in range(b.orbitals[0], b.orbitals[1]+1):
                            bond_order += 2 * matrix[0][p,q] * matrix[1][p,q] 

                    print(' bond {:>8} - {:<8}  {:<8.3f}'.format(a.id+'('+str(i+1)+')', b.id+'(' + str(j+i+2)+')', bond_order))
                      
    if mode == 'spin':
        #alpha and beta density
        print('\n charges\n atom    ',matrix[4][0], ' density    ',matrix[4][1], \
              ' density     ', matrix[4][0]+'+'+matrix[4][1], 'density       charge')
        density = np.zeros((len(atoms), 2))
        for i, a in enumerate(atoms):
            for j in range(a.orbitals[0], a.orbitals[1]+1):
                density[i][0] += matrix[3][0][j,j] ; density[i][1] += matrix[3][1][j,j]
            print('  {:2}        {:>5.2f}          {:>5.2f}            {:>5.2f}          {:>5.2f}'.format(getSymbol(a.number), \
                                                            density[i][0], density[i][1], (density[i][0]+density[i][1]), \
                                                            getValenceElectrons(a.number)-(density[i][0]+density[i][1])))
        #bond order
        from atom import isBond
        print('\n inferred bonds by covalent radii')
        for i, a in enumerate(atoms):
            for j, b in enumerate(atoms[i+1:]):
                if isBond(atoms,i, j+i+1): 

                    alpha_bond_order = 0.0 ; beta_bond_order = 0.0
                    for p in range(a.orbitals[0], a.orbitals[1]+1):
                        for q in range(b.orbitals[0], b.orbitals[1]+1):
                            alpha_bond_order += 2.0 * matrix[3][0][p,q] * matrix[1][p,q] 
                            beta_bond_order  += 2.0 * matrix[3][1][p,q] * matrix[1][p,q] 

                    print(' bond {:>8} - {:<8}  ({:1}) {:<8.3f} ({:1}) {:<8.3f} ({:3}) {:<8.3f}'. \
                          format(a.id+'('+str(i+1)+')', b.id+'(' + str(j+i+2)+')', matrix[4][0], alpha_bond_order,matrix[4][1], \
                                 beta_bond_order, matrix[4][0]+'+'+matrix[4][1],alpha_bond_order+beta_bond_order))

        #spin density matrix
        spin_density = matrix[3][0] - matrix[3][1]
        report('spin density', matrix=spin_density)

        #spin density and hyperfine coupling constant
        print('\n atom   spin density    hyperfine coupling constant')
        for i, a in enumerate(atoms):
            hpf = 0.0
            if parameters['method'] == 'indo': hpf = getData('ciso')[a.number-1] * spin_density[a.orbitals[0], a.orbitals[0]]
            sd = spin_density[a.orbitals[0],a.orbitals[0]]
            print('  {:2}     {:>8.4f}               {:>8.4f}'.format(i, sd, hpf))
