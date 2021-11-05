from cindo import scf
import numpy as np
from math import log10

#these are values taken from references in 'harpy(se).md'. 'oh-'' and co(cndo) are Colby College program otherwise ja-inst reference.
ccndo = {'h2': [-1.4746672338, -0.7650, 0.2375], \
         'LiH': [-1.0877947131, -0.4856, 0.0332, 0.0773, 0.0773, 0.2215], \
         'h2o':[-19.8337964306, -1.5109, -0.7585, -0.7178, -0.6534, 0.2920, 0.3750], \
         'co2': [-43.6006314529, -1.6663, -1.6156, -0.9100, -0.9100, -0.8975, -0.7512, -0.5771, 0.1890, 0.1890, 0.3089, 0.5785], \
         'oh-': [-18.9058, -0.7254, -0.0936, 0.0221, 0.0221, 0.9626 ], \
         'co':[-25.0618, -1.6156, -0.8987, -0.7369, -0.7369, -0.6444, 0.1616, 0.1616, 0.4687 ]}
cindo = {'h2': [-1.4746672338, -0.7650, 0.2375], \
         'co': [-23.9554487448, -1.6485, -0.8328, -0.7226, -0.7226, -0.5718, 0.1661, 0.1661, 0.4245], \
         'LiH': [-1.0808647966, -0.4769, 0.0163, 0.0774, 0.0774, 0.2335], \
         'LiF':[-26.5572392104, -1.2902, -0.4357, -0.4357, -0.4289, 0.0011, 0.0777, 0.0777, 0.1595], \
         'n2':[-22.0894495119, -1.6021, -0.8443, -0.7112, -0.7112, -0.5999, 0.1898, 0.1898, 0.4498]}
ocndo = {'BeF':[-28.8701062969, -1.5331, -0.6707, -0.6554, -0.6554, -0.3814, 0.0850, 0.0850, 0.2443, \
                                -1.5311, -0.6554, -0.6554, -0.6543, -0.0296, 0.0850, 0.0850, 0.2485],
         'b2':[-6.8525955500, -1.0742, -0.5960, -0.5960, -0.4951, 0.0711, 0.3019, 0.3019, 0.4989]}
oindo = {'BeF':[-27.5618358586, -1.5366, -0.6525, -0.6297, -0.6297, -0.3890, 0.0729, 0.0729, 0.2258, \
               -1.5316, -0.6344, -0.6344, -0.6338, -0.0107, 0.1096, 0.1096, 0.2514], \
         'b2':[-6.6521988910, -1.0911, -0.5972, -0.5816, -0.5388, 0.0473, 0.3007, 0.3164, 0.5019, \
               -1.0514, -0.5518, -0.1542, -0.0687, 0.1481, 0.3462, 0.3947, 0.5214]}

#verify CNDO closed shell molecules
print(' running tests for CNDO closed shell')

cndo_closed = ['h2', 'LiH', 'h2o', 'co2', 'oh-', 'co']

for molecule in cndo_closed:

    e, eigen = scf(section = molecule, silent=True, method='cndo')

    verified = False
    computed = [e] 
    agreement_level = '' 

    for i in eigen:
        computed.append(i)

    if molecule == 'h2':
        reference = [-1.4746672338082725, -0.76495263, 0.23749213]
    elif molecule == 'LiH':
        reference = [-1.0877945806496334, -0.48565296,  0.03318167,  0.0773299, 0.0773299 ,0.2216131 ]
    elif molecule == 'h2o':
        reference = [-19.833796430480557, -1.51087429, -0.75858144, -0.71773104, -0.65334253, 0.29207437, 0.37492905]
    elif molecule == 'co2':
        reference = [-43.600631452868726, -1.66627496, -1.61554049, -0.91003531, -0.91003531, -0.89751095, -0.75125262, \
                                         -0.57702742, -0.57702742,  0.18901019,  0.18901019,  0.30890575,  0.57847806]
    elif molecule == 'oh-':
        reference = [-18.905826671031107, -0.72607302, -0.09339874, 0.02158677, 0.02158677, 0.96251903]
    elif molecule == 'co':
        reference = [ -25.06193976955334, -1.61129411, -0.89487624, -0.73317177, -0.73317177, -0.64520116, 0.16145653, \
                                           0.16145653,  0.46961989]
    else: 
        pass

    if (abs(reference[0] - ccndo[molecule][0]) <= 1e-6): agreement_level = '\u2713'

    verified = np.allclose(reference, computed, rtol=1e-8, atol=1e-8)
    print('{:<6}......  {:6}... {:<1}'.format(molecule, ['Failed', 'Passed'][verified], agreement_level))

#verify INDO closed shell molecules
print('\n running tests for INDO closed shell')

indo_closed = ['h2','co','LiH', 'LiF','n2']

for molecule in indo_closed:

    e, eigen = scf(section = molecule, silent=True, method='indo')

    verified = False
    computed = [e]
    agreement_level = ''

    for i in eigen:
        computed.append(i)

    if molecule == 'h2':
        reference = [-1.4746672338082725, -0.76495263, 0.23749213]
    elif molecule == 'co':
        reference = [-23.955448744884194, -1.64915565, -0.83350462, -0.72328612, -0.72328612, -0.57171936,  0.16610008, \
                                           0.16610008,  0.4243893 ]
    elif molecule == 'LiH':
        reference = [-1.080864796645447, -0.47687018, 0.01634097,  0.07742846,  0.07742846,  0.23346346]
    elif molecule == 'LiF':
        reference = [-26.55723937401671, -1.29027553, -0.435813948, -0.435813948, -0.428907103, 0.00111064486,  0.0777006325, \
                      0.0777006325,  0.159505202]
    elif molecule == 'n2':
        reference = [-22.089449534162505, -1.6021414, -0.8443456, -0.71117576, -0.71117576, -0.59989881,  0.18982791, 0.18982791, 0.44981407]
    else:
        pass

    if (abs(reference[0] - cindo[molecule][0]) <= 1e-6): agreement_level = '\u2713'

    verified = np.allclose(reference, computed, rtol=1e-8, atol=1e-8)
    print('{:<6}......  {:6}... {:<1}'.format(molecule, ['Failed', 'Passed'][verified], agreement_level))

#verify CNDO open shell molecules
print('\n running test for CNDO open shell')

cndo_open = ['BeF','b2']

for molecule in cndo_open:

    e, alpha, beta = scf(section = molecule, silent=True, method='cndo')

    verified = False
    computed = [e]
    agreement_level = ''

    for i in alpha:
        computed.append(i)
    for i in beta:
        computed.append(i)

    if molecule == 'BeF':
        reference = [-28.870106547100523, -1.53308675, -0.67072603, -0.65551103, -0.65551103, -0.38142462, 0.0849626, 0.0849626, 0.24429734, \
                     -1.53116828, -0.65551103, -0.65551103, -0.65409399, -0.02958809, 0.0849626, 0.0849626, 0.24854705]
    elif molecule == 'b2':
        reference = [-6.852595772821855, -1.07419081, -0.5960284, -0.5960284, -0.4950937, 0.07107642, 0.30194534, 0.30194534, 0.49893797, \
                     -1.06823691, -0.5960284, -0.18536705, -0.083563, 0.12672679, 0.30194534, 0.36354962, 0.49806863]
    else:
        pass

    if (abs(reference[0] - ocndo[molecule][0]) <= 1e-6): agreement_level = '\u2713'

    verified = np.allclose(reference, computed, rtol=1e-8, atol=1e-8)
    print('{:<6}......  {:6}... {:<1}'.format(molecule, ['Failed', 'Passed'][verified], agreement_level))

#verify CNDO open shell molecules
print('\n running test for INDO open shell')

cndo_open = ['BeF','b2']

for molecule in cndo_open:

    e, alpha, beta = scf(section = molecule, silent=True, method='indo')

    verified = False
    computed = [e]
    agreement_level = ''

    for i in alpha:
        computed.append(i)
    for i in beta:
        computed.append(i)

    if molecule == 'BeF':
        reference = [-27.56183584677248, -1.53630347, -0.65235493, -0.62947444, -0.62947444, -0.38906385,  0.0727967, 0.0727967, 0.2255306 , \
                     -1.53114376, -0.63421838, -0.63421838, -0.63325775, -0.01075935, 0.10949617, 0.10949617, 0.25118423]
    elif molecule == 'b2':
        reference = [-6.65219892381821, -1.09112702, -0.59723657, -0.58158737, -0.53877044,  0.04734231,  0.30073718, 0.31638638,  0.5019444, \
                     -1.05138446, -0.55180617, -0.15418582, -0.06870129, 0.14807536, 0.34616758, 0.39473085, 0.52140795]
    else:
        pass

    if (abs(reference[0] - oindo[molecule][0]) <= 1e-6): agreement_level = '\u2713'

    verified = np.allclose(reference, computed, rtol=1e-8, atol=1e-8)
    print('{:<6}......  {:6}... {:<1}'.format(molecule, ['Failed', 'Passed'][verified], agreement_level))
