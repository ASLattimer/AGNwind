import numpy as np 
setnum = 0 # 0 is a placeholder
ntrial = 64 # number of runs
rad_steps = 500 # number of steps in the radial grid
t_grid = 10**np.linspace(-25,15,num=100,endpoint=True) # default number of points = 50

RandParam_array = np.load('set{}/param_array.npy'.format(setnum),mmap_mode='r') 
Mbh = RandParam_array[runnum][0] 
fwind = RandParam_array[runnum][1] 
shield = RandParam_array[runnum][2] 
alpha = RandParam_array[runnum][3] 
rL = RandParam_array[runnum][4] 
uwind = RandParam_array[runnum][5] 
mdot = RandParam_array[runnum][6] 
floor_frac = RandParam_array[runnum][7] 

import glob
import sys
from scipy import optimize

import traceback
import periodictable
from periodictable import *
from scipy import interpolate
from astropy import units as u

import warnings ## suppress warnings
warnings.filterwarnings("ignore")

import pyagn
import pyagn.sed # get sed class file
from pyagn.sed import SED # get SED function from sed file

# import constants and pre-defined functions
from FunkyParms_Phase2 import *

## OTHER INITIAL PARAMETERS ------------------------------------------------------------------------------------------------------------------------------------------
Mbh_g = Mbh*xMsun # black hole mass in g
astar = 0 # black hole dimensionless spin absolute value
hard_xray_frac = 0.05
temp_floor = 1000 # define a minimum temperature

## GENERATE THE SED OF THE SOURCE
sed_model = SED(M=Mbh, mdot=mdot, astar=astar,reprocessing=True)
nu = sed_model.freq_range
np.save('set{}/run{}/nu.npy'.format(setnum,runnum), nu) # save nu for later

# DEFINE SOME PROPERTIES
gravity_rad = sed_model.gravity_radius # self gravity radius in Rg 
gravity_rad_cm = sed_model.gravity_radius*Rg(Mbh) # self gravity radius in CM
corona_rad = sed_model.corona_radius # corona radius in Rg

#RADIAL GRID (distances in cm unless otherwise specified)
rad_in = sed_model.corona_radius*Rg(Mbh)  #sed_model.isco*Rg(Mbh) 
rad_out = (3*u.kpc).to(u.cm).value # in cm, constant
router_Rg = rad_out/Rg(Mbh) # outermost radius in Rg
# rad_grid = np.logspace(np.log10(rad_in),np.log10(rad_out),num=rad_steps,endpoint=True) #original rad grid, evenly spaced in logspace
# rad_grid[-1] = rad_out
numb = 2
a = np.log10(rad_in)**(1/numb)
b = np.log10(rad_out)**(1/numb)
rad_grid = 10**(np.linspace(a,b,num=rad_steps,endpoint=True)**numb)
rad_grid[-1] = rad_out
np.save('set{}/run{}/rad_grid'.format(setnum,runnum), rad_grid)

# # save properties
# col1 = ['risco_Rg','risco_cm','gravity_rad','gravity_rad_cm','corona_rad']
# col2 = ['r_isco (Rg)','r_isco (cm)','self gravity radius (Rg)', 'self gravity radius (cm)','corona radius (Rg)']
# col3 = [sed_model.isco,sed_model.isco*Rg(Mbh),gravity_rad,gravity_rad_cm,corona_rad]

# data = np.vstack((col1,col2,col3)).T
# output_name = 'set{}/run{}/properties_{}.txt'.format(setnum,runnum,runnum)
# ascii.write(data,output_name,delimiter='|',format='fixed_width',names=('Variable','Description','Value'),overwrite=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## DEFINE SOME FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------------------

## DEFINE OPACITY
# Total opacity = sum over (n_ion * sigma(nu)) for each ionization state of each element, excluding fully ionized
def opacity(freq,rind):
    ni = xnion_list[rind]
    tot_sum = np.zeros(shape=len(freq))
    for el_ind in range(1,nZmax+1): # this starts at 1 bc ind=0 is always 0
        ions = np.arange(1,el_ind+1)
        bob = np.asarray([ni[el_ind,i]*(np.squeeze(photocross(el_ind,i,freq))) for i in ions])
        el_sum = np.sum(bob,axis=0) # sum over ions for element of Z=el_ind
    
        tot_sum = tot_sum+el_sum
    
    return tot_sum + (xne_list[rind]*sigmaT)

# DEFINE OPTICAL DEPTH
# Optical depth delta_tau = opacity * radial step (thickness of slab, cm)
def delta_tau(freq,rind):
    rad_step = rad_grid[rind+1]-rad_grid[rind]
    optical_depth = opacity(freq,rind) * rad_step
    return optical_depth


# DEFINE ATTENUATION OF THE LUMINOSITY
def next_Lnu(freq,rind):
    # rad_ratio = rad_grid[rind]/rad_grid[rind+1]
    new_Lnu = Lnu_list[rind]*np.exp(-delta_tau(freq,rind))#*(rad_ratio**2)
    return new_Lnu


def next_Lfloor(rind):
    opa = xne_list[rind]*sigmaT
    if rad_grid[rind]<rL:
        opa = opa/shield
    rad_step = rad_grid[rind+1]-rad_grid[rind]
    opt_depth = opa*rad_step
    new_Lfloor = floor_L_list[rind]*np.exp(-opt_depth)
    return new_Lfloor

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# compute total flux
r_isco = sed_model.isco*Rg(Mbh) # isco in cm
eta = sed_model.efficiency
Mdot_wind = fwind*(sed_model.mass_accretion_rate)

corona_rad = sed_model.corona_radius
nL = Mdot_wind/(4*pi*(rL**2)*xmHyd*uwind)

# DEFINE L_NU FOR FIRST RADIAL GRID CELL - including disk, corona, and warm region components
try:
    Lnu_corona_init = sed_model.corona_flux_erg(rad_grid[0])*4*pi*(rad_grid[0]**2)
except:
    print('no corona, run ={}'.format(runnum))
    sys.exit()
Lnu_warm_init = sed_model.warm_flux_erg(rad_grid[0])*4*pi*(rad_grid[0]**2)
Lnu_disk_init = sed_model.disk_flux_erg(rad_grid[0])*4*pi*(rad_grid[0]**2)
Lnu_init = Lnu_disk_init + Lnu_corona_init + Lnu_warm_init

# floor_flux_init = floor_frac * sed_model.total_flux_erg(rad_grid[0])

mean_ionfrac = np.zeros(shape=(len(rad_grid),nZmax+2))
ion_array=np.arange(0,nZmax+2)

## BEGIN LOOP #------------------------------------------------------------------------------------------------------------------------------------------------

## CALCULATE INITIAL TOTAL FLUX
# nL = Mdot_wind/(4*pi*(rL**2)*xmHyd*uwind)
# Nyay_H = (nL*(rL-rad_grid[0]))+((nL*rL)/(alpha-1))
# F0 = sed_model.total_flux_erg(rad_grid[0])
# Fyay = np.multiply(F0,np.exp(-Nyay_H*sigmaT))
# floor_flux_init = floor_frac*Fyay
# LOS_flux = Lnu_init/4/pi/(rad_grid[0]**2) # [erg/s/Hz]/[cm^2] --> [erg/cm^2/s/Hz]
# floor_inds = np.where(LOS_flux<floor_flux_init)[0] # find indices where to implement floor due to disk

# Ftot_init = LOS_flux # define flux at observer (including floor) for the next radial step
# Ftot_init[floor_inds] = floor_flux_init[floor_inds] 
floor_L_init = Lnu_init*floor_frac
floor_flux_init = floor_L_init/4/pi/(rad_grid[0]**2)
Ftot_init = Lnu_init/4/pi/rad_grid[0]**2

nH_list = [] # hydrogen NUMBER density cm^-3
NHyd_list = [] # hydrogen COLUMN density cm^-2
TX_list = []
LX_list = []
T_list = []
xne_list = []
xnion_list = []
xi_list = []
xnel_list = []
Lnu_list = [Lnu_init] # luminosity along primary LOS, includes attenuation
Ftot_list = [Ftot_init] # total flux along total LOS, includes all attenuation and floor
floor_flux_list = [floor_flux_init] # flux along secondary LOS
floor_L_list = [floor_L_init] # luminosity along the secondary LOS
Jnu_list = []
Qall_list = []
#delta_list =[]

rind_rL = np.where(rad_grid>rL)[0][0] # rind where radius is first more than launch radius

## Define L_nu and Ftot for the first cell:
L_nu = Lnu_init
Ftot = Ftot_init

# -----------------------------------IONIZATION BALANCE LOOP OVER RADIAL STEPS------------------------------------------
for rind in range(0,len(rad_grid)): 
    ## properties of this parcel
    rwind  = rad_grid[rind] # radius to the parcel (cm) along the line of sight (LOS)

    Jnu = Ftot/4/pi
    Jnu_list.append(Jnu)

    TX = TXray2(nu,Jnu,rwind) # x-ray temp, calculated from spectrum
    TX_list.append(TX)
    vwind  = u_r(rwind,rL,uwind) # velocity of the wind
    Lx = LX(nu,Jnu,rwind) # hydrogen ionizing luminosity of cell, erg/s
    LX_list.append(Lx)

    nHwind = nH(rwind,rL,nL,shield,alpha) # hydrogen density for radial cell
    
    nH_list.append(nHwind)

    xi = Lx/nHwind/(rwind**2)
    ion_param_log = np.log10(xi)
    xi_list.append(xi)

    if (rwind > rL):     # calculate hydrogen column density
        NH1 = 0.
        NH2 = nL*rwind/(alpha-1.)*((rL/rwind)**alpha)
    if (rwind < rL):
        NH1 = shield * nL * (rL - rwind)
        NH2 = nL*rL/(alpha-1.)

    NHyd_list.append(np.log10(NH1+NH2))

    # -----------------------------------THERMAL EQUILIBRIUM-----------------------------------------------------------------
    #  set up grid of temperatures to get a function we can find root of 
    ntrial = 50000 
    amin   = 2 
    amax   = 20
    Ttrial = 10**np.linspace(amin,amax,num=ntrial,endpoint=True)

    ax_last = 1 - Ttrial/TX
    # ax_last[ax_last < 1] = 0
    
    delta  = 0. # original: delta = 0 --> Qad is a true cooling term
    if rwind < rL: 
    #    delta  = 0. # setting this to zero bc vwind=0 before rL so it doesn't matter anyway
        enn = 0
    elif rwind > rL:
        enn = alpha 

#    delta_list.append(delta) # save temp at each cell for later

    P = 2.*nHwind*boltzk*Ttrial

    LHS = (AC_dyda*8.9e-36*xi*(TX - 4.*Ttrial)) \
        +(AX_dyda*1.5e-21*((xi**0.25)/np.sqrt(Ttrial))*(ax_last)) \
        -(AB_dyda*3.3e-27*np.sqrt(Ttrial)) \
        - (AL_dyda*1.7e-18*np.exp(-TL/Ttrial)*(xi**-1)*(Ttrial**-0.5) + 1e-24) 

    Qrad = (nHwind**2)*LHS 
    Qadi = ((vwind*P)/rwind) * ((1.5*delta) - enn) 
    Qall = Qrad + Qadi

    Qall_list.append(Qall)

    if np.log10(xi)<=-7:
        T = temp_floor
    else:
        # interpolate a function so we can find the root
        f = interpolate.interp1d(Ttrial, Qall)
        try:
            sol = optimize.root_scalar(f, bracket = [Ttrial[0],Ttrial[-1]])
        except Exception:
            # traceback.print_exc()
            print('No thermal equilibrium root found, run#={}, rind={}'.format(runnum,rind) )
            np.save('set{}/run{}/nH.npy'.format(setnum,runnum),nH_list)
            np.save('set{}/run{}/N_Hyd.npy'.format(setnum,runnum),NHyd_list)
            np.save('set{}/run{}/TX'.format(setnum,runnum), TX_list)
            np.save('set{}/run{}/LX'.format(setnum,runnum), LX_list)
            np.save('set{}/run{}/T'.format(setnum,runnum), T_list)
            np.save('set{}/run{}/xne'.format(setnum,runnum), xne_list)
            np.save('set{}/run{}/xnion'.format(setnum,runnum), xnion_list)
            np.save('set{}/run{}/xi'.format(setnum,runnum), xi_list)
            np.save('set{}/run{}/Lnu'.format(setnum,runnum), Lnu_list)
            np.save('set{}/run{}/Ftot'.format(setnum,runnum), Ftot_list)
            np.save('set{}/run{}/meanionfrac'.format(setnum,runnum), mean_ionfrac)
            np.save('set{}/run{}/xn_el.npy'.format(setnum,runnum,rind), xnel_list)
            np.save('set{}/run{}/Jnu.npy'.format(setnum,runnum,rind), Jnu_list)
            np.save('set{}/run{}/Qall.npy'.format(setnum,runnum),Qall_list)
            np.save('set{}/run{}/floor_flux.npy'.format(setnum,runnum),floor_flux_list)
 #           np.save('set{}/run{}/delta'.format(setnum,runnum), delta_list)
            sys.exit() # end the process
        
        if sol.root < temp_floor:
            T = temp_floor
        else:
            T = sol.root

    T_list.append(T) # save temp at each cell for later
    # ----------------------------------------------------------------------------------------------------------------------

    ## Define electron temperature in K for this cell --> from temp equilibrium calc.
    T_e = T 

    # -------------------------------IONIZATION BALANCE--------------------------------------------------------------------
    xn_el = abun_by_n*nHwind # number density for each element, gotten from nH for this cell
    xnel_list.append(xn_el)

    xne = nHwind * 0.8

    niter = 50
    for iiter in range(1,niter): # number of iterations of the loop

        nratio = np.zeros(shape=(nZmax+2,nZmax+2))
        xnion = np.zeros(shape=(nZmax+2,nZmax+2))
        denom = np.ones(shape=(nZmax+2))

        for iel in range(1,nZmax+1):  #range(1,93): iel=iZel
            # Compute relative ionization fractions for each pair of species.
            iion_vec = np.arange(1,iel+1)

            if iel<=28:
                CCC = xne*cfit(iel,iion_vec,T_e)
                DDD = xne*dfit(iel,iion_vec,T_e)
                AAA = xne*aufit(iel,iion_vec,T_e)
            else:
                CCC = xne*cheavy (iel,iion_vec,T_e)
                DDD = 0.0 # set dheavy to 0 bc it's hard-coded for T=10^6 K
                AAA = 0.0 

            if iel<=30:
                # PPP = PI(iel,rwind,nu,L_nu)  
                PPP = PI(iel,nu,Jnu)
                RRR = xne*rrfit(iel,iion_vec,T_e) 
                # TBR = tbr(iel,iion_vec,CCC,xne,T_e)
            else:
                PPP = 0
                # TBR = 0
                RRR = xne*rrheavy(iel,iion_vec,T_e)

            nratio[iel, 1:len(iion_vec)+1] = (CCC+AAA+PPP)/(RRR+DDD)#+TBR)s

            for iion in range(iel, 1-1, -1):
                denom[iel] = denom[iel]* nratio[iel,iion]
                denom[iel] = denom[iel] +1
            xnion[iel,1] = xn_el[iel]/denom[iel]
            for iion in range(2,iel+1+1):
                xnion[iel,iion] = xnion[iel,iion-1] * nratio[iel,iion-1]
            if xnion[iel,1]==0.0 and denom[iel]==np.inf: ## add in caveat for when fully ionized & we hit the computer max float limit
                xnion[iel,iel+1] = xn_el[iel] # all in the fully ionized state

        # Second, recompute electron density    
        xne_new = 0.0
        for iel in range(1,nZmax+1):
            for iion in range(1,iel+1+1):
                xne_new = xne_new + xnion[iel,iion]*(iion-1)
        
        if np.abs(np.log10(xne_new/xne)-1) < 1e-5:
            print(iiter)
            break 
        else:
            xne = np.sqrt(xne_new*xne) # --> undercorrection
    
    if xne<1e-10*nHwind:
        xne = 1e-10*nHwind
    xne_list.append(xne)
    xnion_list.append(xnion)

    ionfrac_array = np.zeros(shape=(nZmax+2,nZmax+2))

    for iel in range(1,nZmax+1):
        ionfrac_array[iel,:] = xnion[iel,:]/xn_el[iel]

    nion_ne = np.zeros(shape=(nZmax+2,nZmax+2))

    for iel in range(1,nZmax+1):
        for iion in range(1,iel+1+1):
            nel_nH_ratio = xn_el[iel]/xn_el[1]  # n_el/n_H ratio,  xn_el[0] = n_el for Hydrogen = n_h
            nH_ne_ratio = xn_el[1]/xne_list[rind]          # n_H/n_e ratio # n_e at this radial step
            nion_ne[iel,iion] = ionfrac_array[iel,iion]*nel_nH_ratio*nH_ne_ratio 

    np.save('set{}/run{}/nion_ne_arrays/nion_ne{}.npy'.format(setnum,runnum,rind), nion_ne) # save this so we can calculate all qs and Ws later

    #------------------------------------CALCULATE NEW LUMINOSITY & FLOOR FOR NEXT RAD STEP--------------------------------------
    if rind<len(rad_grid)-1: # don't need to do this on the last cell
        new_Lnu = next_Lnu(nu,rind)  # define new L_nu (attenuated, w/o floor) of the primary LOS for the next radial step 
        L_nu = new_Lnu
        Lnu_list.append(L_nu) # save Lnu for later

        new_Lfloor = next_Lfloor(rind) # define luminosity of the secondary LOS for the next radial step
        Lfloor = new_Lfloor
        floor_L_list.append(Lfloor) # save Lfloor for later
        
        next_rwind = rad_grid[rind+1]

        floor_flux = Lfloor/4/pi/next_rwind**2 #*floor_frac
        LOS_flux = L_nu/4/pi/next_rwind**2 # [erg/s/Hz]/[cm^2] --> [erg/cm^2/s/Hz]
        floor_inds = np.where(LOS_flux<floor_flux)[0] # find indices where to implement floor due to disk

        Ftot = LOS_flux # define flux at observer (including floor) for the next radial step
        Ftot[floor_inds] = floor_flux[floor_inds] # replace relevant indices with floor flux

        Ftot_list.append(Ftot) # save Ftot for later
        floor_flux_list.append(floor_flux)

    for el_num in range(0,nZmax+1):
        Q = (np.sum(ionfrac_array[el_num]*ion_array)-1)/(el_num)

        mean_ionfrac[rind,el_num] = Q
    
# -----------------------END OF IONIZATION BALANCE LOOP-----------------------------------------------------------------


# -----------------------SAVE EVERYTHING--------------------------------------------------------------------------------
np.save('set{}/run{}/nH.npy'.format(setnum,runnum),nH_list)
np.save('set{}/run{}/N_Hyd.npy'.format(setnum,runnum),NHyd_list)
np.save('set{}/run{}/TX'.format(setnum,runnum), TX_list)
np.save('set{}/run{}/LX'.format(setnum,runnum), LX_list)
np.save('set{}/run{}/T'.format(setnum,runnum), T_list)
np.save('set{}/run{}/xne'.format(setnum,runnum), xne_list)
np.save('set{}/run{}/xnion'.format(setnum,runnum), xnion_list)
np.save('set{}/run{}/xi'.format(setnum,runnum), xi_list)
np.save('set{}/run{}/Lnu'.format(setnum,runnum), Lnu_list)
np.save('set{}/run{}/Ftot'.format(setnum,runnum), Ftot_list)
np.save('set{}/run{}/meanionfrac'.format(setnum,runnum), mean_ionfrac)
np.save('set{}/run{}/xn_el.npy'.format(setnum,runnum,rind), xnel_list)
np.save('set{}/run{}/Jnu.npy'.format(setnum,runnum,rind), Jnu_list)
np.save('set{}/run{}/Qall.npy'.format(setnum,runnum),Qall_list)
np.save('set{}/run{}/floor_flux.npy'.format(setnum,runnum),floor_flux_list)
#np.save('set{}/run{}/delta'.format(setnum,runnum), delta_list)


## CALCULATE q, W, and qW ARRAYS--------------------------------------------------------------------------------------------------------------------------------------
rad_grid = np.load('set{}/run{}/rad_grid.npy'.format(setnum,runnum))
Lnu_list = np.load('set{}/run{}/Lnu.npy'.format(setnum,runnum))
xne_list = np.load('set{}/run{}/xne.npy'.format(setnum,runnum))
T_list = np.load('set{}/run{}/T.npy'.format(setnum,runnum))
nu = np.load('set{}/run{}/nu.npy'.format(setnum,runnum))
xnel_list = np.load('set{}/run{}/xn_el.npy'.format(setnum,runnum))
NH = np.load('set{}/run{}/N_Hyd.npy'.format(setnum,runnum))
xi = np.load('set{}/run{}/xi.npy'.format(setnum,runnum))

# reject the radial steps that are unphysical
rind_mask = (NH<25) & (NH>17) #& (np.log10(xi)<5) & (np.log10(xi)>-5) # might not need xi slices
slice_inds = np.squeeze(np.where(rind_mask==True))

rad_Qbar = np.zeros(shape=(len(rad_grid)))

all_lines_file = 'All_Lines_Array_v2.npy'
array = np.load(all_lines_file,mmap_mode='r') # should already be in correct shape (7,# of lines)

M_t1 = np.zeros(len(rad_grid))

for rind in slice_inds: 
    M_array_path = 'set{}/run{}/M_arrays/M{}.npy'.format(setnum,runnum,rind) 
    # if os.path.exists(M_array_path) == True:
    #     continue
    
    T = T_list[rind] # T is the temp at this radial step
    Lnu = Lnu_list[rind]
    xn_el = xnel_list[rind]

    nion_ne = np.load('set{}/run{}/nion_ne_arrays/nion_ne{}.npy'.format(setnum,runnum,rind))

    ntot = xne_list[-1]+np.sum(xn_el[1:]) # total number density --> need this for partition func

    Qbar = 0

    thing_1 = array[0] ## atomic num
    atom_ind = thing_1[0].astype(int) # define which atom (for naming later)

    thing_2 = array[1] ## ion state
    ion_ind = thing_2[0].astype(int) # define which ionization state (for naming later)

    thing_3 = np.abs(array[2]) ## wavelength
    thing_4 = array[3] ## gf
    thing_5 = array[4] ## E_i (cm^-1)
    
    ## q-VALUES ------------------------------------------------------------------------------------------------
    e_radius = (xmelectron*(clight**2))/(eelectron**2)  ## does not depend on arrays
    prefactor = (3/8)*(thing_3*(10**-8))*e_radius*thing_4
    v_0 = clight/(thing_3*(10**-8))
    E_erg = thing_5 * hPlanck *clight
    exponent = np.exp(-E_erg/(boltzk*T))
    denom =  Upart_fun(thing_1.astype(int),thing_2.astype(int),ntot,T)
    first_frac = exponent/denom
    if first_frac.any() >1.0:
        print('oh no -->{}'.format(runnum))
    exponent_2 = np.exp((-hPlanck*v_0)/(boltzk*T))
    last = 1-exponent_2
    # put it all together
    q = prefactor * first_frac * nion_ne[thing_1.astype(int),thing_2.astype(int)] *last

    q_file_path = 'set{}/run{}/q_arrays/q_all_{}.npy'.format(setnum,runnum,rind)

    compressed_q_array = np.ma.compressed(q)
    np.save(q_file_path,compressed_q_array)

    # W-VALUES -------------------------------------------------------------------------------------------------
    line_freq = clight/(thing_3*(1e-8))
    line_freq_ind = [np.argmin(np.abs(nu-i)) for i in line_freq] # find the closest sed freq to the line freq
    num = line_freq*Lnu[line_freq_ind] # get Lnu at the line freq       
    denom = L_bol(nu,Lnu)
    W = num/denom

    W_file_path = 'set{}/run{}/W_arrays/w_all_{}.npy'.format(setnum,runnum,rind)
    compressed_W_array = np.ma.compressed(W)
    np.save(W_file_path, compressed_W_array) # not saving to save space

    ## qW VALUES ------------------------------------------------------------------------------------------------
    qW_array = np.multiply(q,W)
    qW_file_path = 'set{}/run{}/total_arrays/qW_all_{}.npy'.format(setnum,runnum,rind)

    compressed_qW_array = np.ma.compressed(qW_array)
    np.save(qW_file_path,compressed_qW_array)

    Qbar = np.sum(compressed_qW_array) 

    rad_Qbar[rind] = Qbar

    ## CALCULATE M(t) VALUES -------------------------------------------------------------------------------------------------------------------------------------
    M_array = np.zeros(len(t_grid)) # make an empty array to store our M(t) values

    final_q_array = compressed_q_array
    final_qW_array = compressed_qW_array

    temp = np.load('set{}/run{}/T.npy'.format(setnum,runnum),mmap_mode='r')[rind]

    vth_coeff = np.sqrt(2*boltzk/xmproton)
    vth = vth_coeff*np.sqrt(temp)
    # try:
    #     tau_i = (clight/vth)*final_q_array*t_grid[:,None]
    #     tau_masked = np.ma.masked_less(tau_i,1e-12)
    #     par_num = np.subtract(1,np.exp(-tau_masked))
    #     par_frac = np.divide(par_num,tau_i)
    #     par_frac = np.ma.filled(par_frac,1) # make the parentheses term-->1 when tau_i-->0

    #     mult_pre = np.multiply(par_frac,final_qW_array)
    #     mult_fin = np.sum(mult_pre,axis=1) # no finite disk factor for now
    #     np.save(M_array_path,mult_fin)
    
    # except:
    for tind,t in enumerate(t_grid):
        tau_i = (clight/vth)*final_q_array*t
        tau_masked = np.ma.masked_less(tau_i,1e-12)

        par_num = np.subtract(1,np.exp(-tau_masked))
        par_frac = np.divide(par_num,tau_i)
        par_frac = np.ma.filled(par_frac,1) # make the parentheses term-->1 when tau_i-->0

        mult_pre = np.multiply(par_frac,final_qW_array)
        mult_fin = np.sum(mult_pre) # no finite disk factor for now

        M_array[tind] = mult_fin
        # continue
  
    np.save(M_array_path,M_array)

    ## calculate M(t=1) for each good radius
    t=1
    tau_i = (clight/vth)*final_q_array*t
    tau_masked = np.ma.masked_less(tau_i,1e-12)

    par_num = np.subtract(1,np.exp(-tau_masked))
    par_frac = np.divide(par_num,tau_i)
    par_frac = np.ma.filled(par_frac,1) # make the parentheses term-->1 when tau_i-->0

    mult_pre = np.multiply(par_frac,final_qW_array)
    mult_fin = np.sum(mult_pre) # no finite disk factor for the time being

    M_t1[rind] = mult_fin

M_t1_path = 'set{}/run{}/M_t1'.format(setnum,runnum) 
np.save(M_t1_path,M_t1)
np.save('set{}/run{}/rad_Qbar.npy'.format(setnum,runnum),rad_Qbar)






