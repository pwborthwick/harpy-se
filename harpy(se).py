#run as python 'harpy(se).py', or python 'harpy(se).py' -filename.hpf or 
#python 'harpy(se).py' -section, where section is a molecule name in the 
#project.hpf file eg to run name=LiH use python 'harpy(se).py' -LiH
#python 'harpy(se).py' without a parameter will run the first molecule in
#the project.hpf file.

import cindo
import time
import sys
import numpy as np

t = time.time()

#concatenate arguments harpy(se).py [-<section> |-<file>]
args = ''
for arg in sys.argv:
    args += arg

method = ''
if ('-cndo' in args):
    method = 'cndo' ; args = args.replace('-cndo', '')
elif ('-indo' in args):
    method = 'indo' ; args = args.replace('-indo', '')

#get project file
molFile = 'project.hpf'
if '.hpf' in args:
    sgra = args[::-1]
    i = sgra.find('fph.')
    j = sgra.find('-', i)
    molFile = sgra[i:j][::-1]
    print('\n',round(cindo.scf(file=molFile, method=method)[0],4), ' in ',round(time.time()-t,3),'s')


elif '-' in args:
    section = args.replace('harpy(se).py','').replace('-','',1)
    print('\n',round(cindo.scf(section=section, method=method)[0],4), ' in ',round(time.time()-t,3),'s')

else:
    print('\n',round(cindo.scf(file=molFile, method=method)[0],4), ' in ',round(time.time()-t,3),'s')