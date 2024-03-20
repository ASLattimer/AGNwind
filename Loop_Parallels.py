# code to set up parallelization of the line of sight calculations
import time
from periodictable import *
import multiprocessing
import os
import glob
import random
from astropy import units as u

from Initial_Parms import *

setnum = 0
ntrial = 64 # number of runs
script_orig = 'LOSCalcs_{}.py'.format(setnum) # change this depending on version of LOSCalcs_MC file

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# CREATE FILE STRUCTURE TO HOLD EACH RUN
Main_Dir = 'set{}/'.format(setnum)
if not os.path.exists(Main_Dir):
    os.mkdir(Main_Dir)

for n in range(ntrial):
    RunDir = 'set{}/run{}'.format(setnum,n) 
    if not os.path.exists(RunDir):
        os.mkdir('set{}/run{}'.format(setnum,n))
        os.mkdir('set{}/run{}/q_arrays/'.format(setnum,n))
        os.mkdir('set{}/run{}/W_arrays/'.format(setnum,n))
        os.mkdir('set{}/run{}/nion_ne_arrays/'.format(setnum,n))
        os.mkdir('set{}/run{}/total_arrays/'.format(setnum,n))  
        os.mkdir('set{}/run{}/M_arrays/'.format(setnum,n))  
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
## CREATE RANDOM PARAMETER ARRAYS (to choose our initial values from)

# Define grids of reasonable values
Mbh_grid = np.logspace(6,11,num=1000, endpoint=True)
fwind_grid = np.logspace(-2, 1,num=1000,endpoint=True) # orig lims: -1, -1.5
S_grid = np.logspace(0,2,num=1000,endpoint=True) 
alpha_grid = np.linspace(1.01, 2., num=1000, endpoint=True)
rL_grid = np.linspace(100,1000,num=1000,endpoint=True) # in gravitational radii
uwind_grid = np.logspace(np.log10((100*u.km/u.s).to(u.cm/u.s).value),np.log10((70000*u.km/u.s).to(u.cm/u.s).value),num=10000,endpoint=True)
mdot_grid = np.logspace(-1.5,1,num=1000, endpoint=True) 
floor_frac_grid = np.logspace(-8,-2,num=100, endpoint=True) # I think 1e-10 is too low... change lower lim to 1e-8

RandParam_array = np.zeros(shape=(ntrial,8))
for n in range(ntrial): # create a master array of all the random initial param values
    Mbh = random.choice(Mbh_grid)                  # mass of black hole in solar masses        
    RandParam_array[n][0] = Mbh
    
    fwind = random.choice(fwind_grid)              # wind fraction
    RandParam_array[n][1] = fwind
    
    shield = random.choice(S_grid)                 # shielding factor
    RandParam_array[n][2] = shield
    
    alpha = random.choice(alpha_grid)              # density exponent
    RandParam_array[n][3] = alpha
 
    rL = random.choice(rL_grid)                    # launch radius (not r_isco), in cm
    RandParam_array[n][4] = rL*Rg(Mbh)

    uwind = random.choice(uwind_grid)              # wind velocity
    RandParam_array[n][5] = uwind
   
    mdot = random.choice(mdot_grid)                # Eddington ratio --> black hole *accretion* rate in Eddington units
    RandParam_array[n][6] = mdot

    floor_frac = random.choice(floor_frac_grid)    # fraction that determines floor
    RandParam_array[n][7] = floor_frac               

## SAVE THE PARAMETER ARRAY -----------------------------------------------------------------------------------------------------------------------------
np.save('set{}/param_array'.format(setnum),RandParam_array)

## MAKE THE SCRIPTS TO RUN  -----------------------------------------------------------------------------------------------------------------------------
for n in range(ntrial):
    run_script = 'LOSCalcs_{}_run{}.py'.format(setnum,n)
    with open(script_orig,'r') as contents:
        main_text = contents.read()
    with open(run_script,'w') as contents:
        contents.write('runnum = {} #run number \n'.format(n))
        contents.write(' \n')
        contents.write(main_text)

# Creating the list of all the scripts
all_scripts = glob.glob('LOSCalcs_{}_run*.py'.format(setnum))

# function to call the script from command line                                                                            
def execute(script):                                                             
    os.system(f'python {script}')                                       

## DO THE RUNS IN PARALLEL --------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    start = time.perf_counter()
    processes = []
    for script in all_scripts: # spawn each process
        p = multiprocessing.Process(target=execute, args=(script,))
        p.start()
        processes.append(p) 

    for p in processes:
        p.join()
    end = time.perf_counter()

    print('All Processes Completed!')
    print(f'Finished in {round(end-start, 2)} second(s)')  
