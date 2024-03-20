# initial parameters and function defitions
import numpy as np
import math
import scipy
from astropy import constants as const
from astropy import units as u
from astropy.io import ascii
import periodictable
from periodictable import *

## DEFINE PARAM GRID MINS AND MAXS
mbh_min = 6 # min value for Mbh grid, log
mbh_max = 11 # max value for Mbh grid, log
fwind_min = -2 # max value for fwind grid, log
fwind_max = 1 # min value for fwind grid, log
S_min = 0 # min value for S grid, log
S_max = 2 # max value for S grid, log
alpha_min = 1.01
alpha_max = 2
rL_min = 100 # min value for rL grid, gravitational radii
rL_max = 1000 # max value for rL grid, gravitational radii
uwind_min = np.log10((100*u.km/u.s).to(u.cm/u.s).value)  # min value for uwind grid, log
uwind_max = np.log10((70000*u.km/u.s).to(u.cm/u.s).value) # max value for uwind grid, log
mdot_min = -1.5 # min value for mdot grid, log
mdot_max = 1 # max value for mdot grid, log
ffloor_min = -8 # min value for floor_frac grid, log
ffloor_max = -2 # max value for floor_frac grid, log

## DEFINE CONSTANTS 
#(all cgs... all the time)

Gconst     = 6.6732e-08 # cm^3 g^-1 s^-2
xMsun      = 1.989e+33 # grams
xLsun      = 3.826e+33
boltzk     = 1.380622e-16
xmHyd      = 1.67333e-24
stefan     = 5.66961e-5
clight     = 2.997925e+10 # cm s^-1
hPlanck    = 6.6260755e-27 # erg*s
hPlanck_eV = 4.135667696e-15 #eV s
xmelectron = const.m_e.to('g').value # mass of electron
eelectron  = 4.80325e-10 # electric charge (cgs) --> cgs units not in constants module
xmproton   = const.m_p.to('g').value # mass of proton
pi         = math.pi
parsec  = 3.085678e+18 # cm

Ryd = 241799050402293 # Hz conversion of 1 Rydberg (13.6eV)
ryd1_Hz = (1/hPlanck_eV)*13.6  
ryd1000_Hz = (1/hPlanck_eV)*13.6*1000 

# Thomson scattering cross section: 
re = (eelectron**2)/xmelectron/clight**2
sigmaT = (8*pi*re**2)/3

#  Dyda et al. (2017) constants to fix (used in thermal eq. calc)
AB_dyda = 3.9   # for their "modified Blondin" model
AC_dyda = 1.0
AX_dyda = 1.0
AL_dyda = 1.0
TL      = 1.3e5

# Load the arrays we'll need for ionization/recombination processes functions
cf = np.load('support_files/cifit_data_array.npy')   # cf array made from cifit_data.txt
CH = np.load('support_files/ciheavy_data_array.npy') # CH array made from ciheavy_data.txt
rrec = np.load('support_files/rrec_data_array.npy')  # rrec array made from rrec_data.txt
rnew = np.load('support_files/rnew_data_array.npy')  # rnew array made from rnew_data.txt
fe = np.load('support_files/ferr_data_array.npy')    # fe array made from ferr_data.txt
AD = np.load('support_files/difit_data_array.npy')   # AD array made from difit_data.txt


## file with elemental abundances from Asplund 2020
# abunds_file = 'support_files/asplund_abunds_2020.txt'
abunds_file = 'support_files/asplund_abunds_SS.txt'


# CFIT --> COLLISIONAL IONIZATION ***********************************************************************************************************************************

# subroutine cfit(iz,inn,temp) --> output: c
# SEE:  http://www.pa.uky.edu/~verner/col.html

# This subroutine calculates rates of direct collisional ionization 
# for all ionization stages of all elements from H to Ni (Z=28)
# by use of the fits from G. S. Voronov, 1997, ADNDT, 65, 1
# Input parameters:  iz - atomic number 
#                   inn n- number of electrons from 1 to iz 
#                   temp  - temperature, K
# Output parameter:  c  - rate coefficient, cm^3 s^(-1)


def cfit(iz,ii,temp): #cfit(iz,inn,t,c)
    inn = iz-ii+1
    c=0.0
    if(iz<1 or iz>28):
        return c
    if(inn<1 or inn>iz):
        return c
    
    te=temp*8.617385e-5
    u=cf[iz,inn,0]/te

    c=cf[iz,inn,2]*(1.0+cf[iz,inn,1]*np.sqrt(u))/(cf[iz,inn,3]+u)**u**cf[iz,inn,4]*np.exp(-u)
    
    return c

cfit=np.vectorize(cfit)


# CHEAVY ***************************************************************************************************************************************************************

# subroutine cheavy(iz,inn,temp) --> output: c
# approximate collision rates for heavy (Z>28) atoms and ions
# (see Cranmer 2000)
# Input parameters: iz - atomic number 
#                   inn - number of electrons from 1 to iz 
#                   temp  - temperature, K
# Output parameter: c  - rate coefficient, cm^3 s^(-1)


def cheavy(iz,ii,temp): #cheavy(iz,inn,t,c)
    inn = iz-ii+1
    c=0.0
    if(iz<29 or iz>92):
        return c
    if(inn<1 or inn>iz):
        return c

    te=temp*8.617385e-05
    u=CH[iz,inn,0]/te


    c=CH[iz,inn,2]*(1.0+CH[iz,inn,1]*np.sqrt(u))/(CH[iz,inn,3]+u)*u**CH[iz,inn,4]*np.exp(-u) 

    return c
cheavy=np.vectorize(cheavy)


## RRFIT *************************************************************************************************************************************************************

# subroutine rrfit(iz,inn,temp) --> output: r
# SEE: http://www.pa.uky.edu/~verner/rec.html
#
# Version 3a. August 19, 1996.
# Written by D. A. Verner, verner@pa.uky.edu 
#
# This subroutine calculates rates of radiative recombination for all ions
# of all elements from H through Zn by use of the following fits:
# H-like, He-like, Li-like, Na-like - Verner & Ferland, 1996, ApJS, 103, 467
# Other ions of C, N, O, Ne - Pequignot et al. 1991, A&A, 251, 680,
#    refitted by Verner & Ferland formula to ensure correct asymptotes
# Fe XIV-XV and Fe XVII-XXIII - Arnaud & Raymond, 1992, ApJ, 398, 394
# Other ions of Mg, Si, S, Ar, Ca, Fe, Ni - 
#                      - Shull & Van Steenberg, 1982, ApJS, 48, 95
# Other ions of Na, Al - Landini & Monsignori Fossi, 1990, A&AS, 82, 229
# Other ions of F, P, Cl, K, Ti, Cr, Mn, Co (excluding Ti I-II, Cr I-IV,
# Mn I-V, Co I)        - Landini & Monsignori Fossi, 1991, A&AS, 91, 183
# All other species    - interpolations of the power-law fits
#
# Input parameters:  iz - atomic number 
#                    inn - number of electrons from 1 to iz 
#                    temp  - temperature, K
# Output parameter:  r  - rate coefficient, cm^3 s^(-1)


def rrfit(iz,ii,temp): #rrfit(iz,inn,t,r)
    inn = iz-ii+1
    r=0.0
    if(iz<1 or iz>30):
        print("rfit called with insane atomic number, iz={}".format(iz))
        return r

    if(inn<1 or inn>iz):
        print("rrfit called with insane number elec, inn={}".format(inn))
        return r

    if(inn<=3 or inn==11 or (iz>5 and iz<9) or iz==10):         
        tt=np.sqrt(temp/rnew[iz,inn,2])
        r=rnew[iz,inn,0]/(tt*(tt+1.0)**(1.0-rnew[iz,inn,1])*(1.0+np.sqrt(temp/rnew[iz,inn,3]))**(1.0+rnew[iz,inn,1]))
        # should one of the rnew[iz,inn,1] indices be rnew[iz,inn,2]?
        
    else:
        tt=temp*1.0e-04
        if(iz==26 and inn <= 13): 
                r=fe[0,inn]/tt**(fe[1,inn]+fe[2,inn]*np.log10(tt)) 
        else:
            r=rrec[iz,inn,0]/tt**rrec[iz,inn,1]

    return r
rrfit=np.vectorize(rrfit)


#RRHEAVY *************************************************************************************************************************************************************

# subroutine rrheavy (iz,inn,temp) --> output: r
# approximate radiative recombination rates for heavy (Z>28) atoms and ions
# (see Cranmer 2000)
# 
# Input parameters:  iz - atomic number 
#                    inn - number of electrons from 1 to iz 
#                    temp  - temperature, K
# Output parameter:  r  - rate coefficient, cm^3 s^(-1)


def rrheavy(iz,ii,temp):#rrheavy(iz,inn,t,r)
    inn = iz-ii+1
    r=0.0
    if(iz<28 or iz>92):
        print("rrheavy called with insane atomic number, iz={}".format(iz))
        return r

    if(inn<1 or inn>iz):
        print("rrheavy called with insane number elec, inn={}".format(inn))
        return r

    ee = iz
    zz = iz-inn

    c0 = -28.25 - 0.04923*ee
    c1 =  2.175 + 0.02212*ee
    aa = np.exp(c0) * (zz+1.0)**c1

    r  = aa / (temp*1.0e-4)**0.8

    return r
rrheavy=np.vectorize(rrheavy)


## DFIT ***********************************************************************************************************************************************************

# subroutine dfit (iz,inn,temp) --> output: alphadi
# by S. Cranmer, but data is from:
# Mazzotta, P., Mazzitelli, G., Colafrancesco, S., and Vittorio, N.
#    1998, A&A Suppl Ser, 133, 403-409.  

# AD(1,iz,ii) : fit parameter c1 (in cm^3/s)
# AD(2,iz,ii) : fit parameter c2 (in cm^3/s)
# AD(3,iz,ii) : fit parameter c3 (in cm^3/s)
# AD(4,iz,ii) : fit parameter c4 (in cm^3/s)
# AD(5,iz,ii) : fit parameter E1 (in eV)
# AD(6,iz,ii) : fit parameter E2 (in eV)
# AD(7,iz,ii) : fit parameter E3 (in eV)
# AD(8,iz,ii) : fit parameter E4 (in eV)

# alphadi = T^(-3/2) * SUM[j=1,4] (cj * exp(-Ej/T))    (T in eV)

# Input parameters:  iz      - atomic number 
#                    inn     - number of electrons from 1 to iz 
#                    ii      - ion number (ii = (iz-in)+1)
#                    temp       - temperature, K
# Output parameter:  alphadi - dielectronic recombination rate in cm^3/s


def dfit(iz,ii,temp): #dfit(iz,inn,t,alphadi)
    inn = iz-ii+1
    alphadi = 0.0
#     ii = iz-inn+1

    if (iz<1 or iz>28):
        return alphadi
    if (inn<1 or inn>iz):
        return alphadi
        
    if (AD[iz,ii,0]==0.0):
        return alphadi

    te      = temp*8.617385e-05
    alphadi = te**(-1.5) * ( AD[iz,ii,0]*np.exp(-AD[iz,ii,4]/te) + AD[iz,ii,1]*np.exp(-AD[iz,ii,5]/te) +
                AD[iz,ii,2]*np.exp(-AD[iz,ii,6]/te) + AD[iz,ii,3]*np.exp(-AD[iz,ii,7]/te))

    return alphadi
dfit=np.vectorize(dfit)


# ## DHEAVY ***********************************************************************************************************************************************************8

# # subroutine dheavy (iz,inn,temp) --> output: d
# #approximate dielectronic recomb. rates for heavy (Z>28) atoms and ions
# #(see Cranmer 2000)

# # IMPORTANT:  THIS ROUTINE IS FOR ONLY T=1.0e6 K !!!!!!!
# # 
# #Input parameters:  iz - atomic number 
# #                   in - number of electrons from 1 to iz 
# #                   temp  - temperature, K
# #Output parameter:  d  - rate coefficient, cm^3 s^(-1)

# def dheavy(iz,ii,temp): #dheavy(iz,inn,t,d)
#     inn = iz-ii+1

#     d=0.0 # this was orginially r=0.0 but I think it should be d=0.0

#     if(temp!=1e6):
#         print("temperature is not 10^6 K !",iz)
#         return d

#     if(iz<28 or iz>92):
#         print("dheavy called with insane atomic number, iz={}".format(iz))
#         return d

#     if(inn<1 or inn>iz):
#         print("dheavy called with insane number elec, inn={}".format(inn))
#         return d

#     d  = 1.26e-12*(CH[iz,inn,0]**0.522)

#     return d



## Define the EXPINT Functions ***************************************************************************************************************************************

def EN(X): # exponential integral
    return scipy.special.exp1(X)
    

def E0(X):
    out = np.exp(-X)/X
    return out

def EXPENX(X):
    if X<=100:
        out = np.exp(X)*EN(X)
    if X>100:
        out = (1./X) - (1./X/X) + (2./X/X/X)
    return out

def IERR(X,N):
    if (X<0 or N<0):
        ierr = -1
        return ierr
    
    elif (EN(X)==-np.inf or EXPENX(X)==-np.inf):
        ierr = 1
    
    elif (EN(X)==np.inf or EXPENX(X)==np.inf or isinstance(N, int)==False):
        ierr = 2
        return ierr
    
    else:
        ierr = 0
        return ierr


# AUFIT **********************************************************************************************************************************************************

# subroutine aufit(iz,inn,temp) --> output: a
# autoionization rates from Landini & Monsignori Fossi (1990)
# 
# Input parameters:  iz - atomic number 
#                    in - number of electrons from 1 to iz 
#                    ii - ion number (ii = iz-inn+1)
#                    temp  - temperature, K
# Output parameter:  a  - rate coefficient, cm^3 s^(-1)

AU = np.load('support_files/auifit_data_array.npy') ## MOVE THIS TO TOP

def aufit(iz,ii,temp):
    inn = iz-ii+1

    a       = 0.0
    # ii      = iz-inn+1
    if (iz<1 or iz>28):
        return a
    if (inn<1 or inn>iz):
        return a
    if AU[iz,ii,0]==0.0:
        return a
    
    
    y = AU[iz,ii,1]/temp
    # RN = 1.0
    N = 1 # needs to be an integer #if this needs to vary I will need to move this to outside the function
    
    if (IERR(y,N) !=0):
        print('error in EXPINT', IERR(y,N),'y=',y,'N=',N)
        return IERR(y,N)
    #endif

    f1 = EXPENX(y) #EXPE1y

    if (inn==3):
        fy    = (2.22*f1) + 0.67*(1.0 - y*f1) + (0.49*y*f1) + 1.20*y*(1.0 - y*f1)
    elif ((inn==11)and(iz<=16)):
        fy    = 1.0 - y*f1
    elif ((inn==11)and(iz>=18)):
        fy    = 1.0 - 0.5*(y - y*y + y*y*y*f1)
    elif ((inn>12)and(inn<=18)and(iz>=18)):
        fy    = 1.0 - 0.5*(y - y*y + y*y*y*f1)
    else:
        fy    = 1.0 + f1
    #endif

    a = AU[iz,ii,0] / np.sqrt(temp) * np.exp(-y) * fy

    return a
aufit=np.vectorize(aufit)



## DEFINE PHOTOIONIZATION CONTRIBUTION ******************************************************************************************************************************
tau_nu = 0.0 # this will need to be refined as per Steve's notes

## Load the Look-Up Table
LUT = np.load('LUT.npy',allow_pickle=True)

# cross_section_file2 = 'Verner1995_photocross' 
n95_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=3)
l95_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=4) 
Eth95_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=5)  # eV
E095_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=6)   #eV
sig095_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=7) # Mb (1e-18 cm^2)
ya95_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=8)
P95_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=9)
yw95_tot = np.genfromtxt('support_files/Verner1995_photocross',comments='%',skip_header=1,usecols=10)

# def PI_by_ion(Z,iion,r,freq,Lnu): 
def PI_by_ion(Z,iion,freq,J): 

    shell_inds = LUT[Z][iion]

    n95 = n95_tot[shell_inds]
    l95 = l95_tot[shell_inds]
    Eth95 = Eth95_tot[shell_inds]  # eV
    E095 = E095_tot[shell_inds]   #eV
    sig095 = sig095_tot[shell_inds] # Mb (1e-18 cm^2)
    ya95 = ya95_tot[shell_inds]
    P95 = P95_tot[shell_inds]
    yw95 = yw95_tot[shell_inds]

    E = hPlanck_eV*freq

    Q = 5.5+l95-(0.5*P95)
    Y = np.array([energy/E095 for energy in E])

    Fy = ((Y-1)**2+(yw95**2))*(Y**-Q)*(1+np.sqrt(Y/ya95))**(-P95)
    sigma_Mb = sig095*Fy # gives sigma_nu in Mb (lol Verner why u do dis)
    sigma_cm = sigma_Mb*1e-18

    for j in range(sigma_cm.shape[1]):
        for i in range(0,len(freq)):
            if E[i]<Eth95[j]:
                sigma_cm[i,j] = 0

    fin_sig_array = np.sum(sigma_cm,1) 

    # Jnu definition------------------------------------------------------------------
    # frac1 = 1/(4*pi) 
    # frac2 = np.divide(Lnu,4*pi*(r**2)) #L_nu/(4*pi*(r**2))
    # # expo = np.exp(-tau_nu) # tau_nu is defined as 0 for now
    # Jnu = frac1*frac2#*expo

    # Calculate PI rate---------------------------------------------------------------  
    # num = 4*pi*Jnu
    num = 4*pi*J
    denom = np.multiply(hPlanck,freq) # hPlanck*nu
    integrand = np.multiply(num/denom,fin_sig_array)
    integral = np.trapz(integrand,freq)
       
    return integral

# def PI(Z,r,freq,Lnu):
#     ions = np.arange(1,Z+1)
#     PP = [PI_by_ion(Z,iion,r,freq,Lnu) for iion in ions]
    
#     return np.asarray(PP)

def PI(Z,freq,J):
    ions = np.arange(1,Z+1)
    PP = [PI_by_ion(Z,iion,freq,J) for iion in ions]
    
    return np.asarray(PP)


## DEFINE 3-BODY RECOMBINATION CONTRIBUTION  **********************************************************************************************************************

## We need the partition functions (taken from Phase 1)
## STATISTICAL WEIGHT FILE
# FILE01 = 'support_files/chianti_ip_statwt.dat'
FILE01 = 'support_files/chianti_ip_statwtV2.dat'
# For 'chianti_ip_statwt.dat':
# Column 1 = atomic number
# Column 2 = ionization level
# Column 3 = ionization potential #check these units
# Column 4 = g_i ground state statistical weight

## PARTITION FUNCTION PARAMETERS
FILE02 = 'support_files/cardona_partition.dat'
# For 'cardona_partition.dat':
# Column 1 = Z
# Column 2 = J 
# Column 3 = epsilon
# Column 4 = G
# Column 5 = m

nZmax  = 30 # maximum atomic number to be considered
RydK   = 157806.515625 # 

# We create arrays of zeros with dimensions (nmax+2 x nmax+2) 
chiK     = np.zeros(shape=(nZmax+2,nZmax+2))
gground  = np.zeros(shape=(nZmax+2,nZmax+2))
Ecardona = np.zeros(shape=(nZmax+2,nZmax+2))
Gcardona = np.zeros(shape=(nZmax+2,nZmax+2))
mcardona = np.zeros(shape=(nZmax+2,nZmax+2))

npt1 = ascii.read(FILE01,delimiter=' ',guess=True)

for i in range(0,len(npt1)):
    i1 = npt1[i][0]  # i1 = atomic number
    i2 = npt1[i][1]  # i2 = ionization level
    x3 = npt1[i][2]  # ionization potential 
    x4 = npt1[i][3]  # ground stat statistical weight
    chiK[i1,i2] = x3 *1.4388064  # Chi = ionization potential of ionization state j
    gground[i1,i2] = x4 # ground stat statistical weight

npt2 = ascii.read(FILE02,delimiter=' ',guess=True)

for j in range(0,len(npt2)):
    i1 = npt2[j][0]  # i1 = Z
    i2 = npt2[j][1]  # i2 = J
    x3 = npt2[j][2]  # x3 = epsilon
    x4 = npt2[j][3]  # x4 = G
    x5 = npt2[j][4]  # x5 = m
    Ecardona[i1,i2+1] = x3 # epsilon
    Gcardona[i1,i2+1] = x4 # G
    mcardona[i1,i2+1] = x5 # m
    
## estimate Cardona parameters for Z>20 (based on correlations found from given parameters for Z<20)
for i in range(0,nZmax+2):
    for j in range(0,nZmax+2):
        if Ecardona[i,j] == 0.:
            Ecardona[i,j] = chiK[i,j] * (0.94631646 - (0.0075761053*i))
        if mcardona[i,j] == 0:
            mcardona[i,j] = 4 * (gground[i,j]**0.79)
        if Gcardona[i,j] == 0:
            Gcardona[i,j] = 113 * (mcardona[i,j] ** 0.66)


#partition function
def Upart_fun(iel,iion,N,T): # this is a function of iion, not in (number of electrons)! 
#     ii = iz-in+1
    # Udummy   = 1. #6.0
    Zeff = iion # assigns effective charge, =/= Z given in table, Z in table is atomic number
    qstar = np.sqrt(Zeff/(2*pi*5.291772e-9)) / (N**(1/6)) # q from Cardona
    ennstar = 0.5*qstar * (1 + np.sqrt(1 + (4/qstar))) # n_* from Cardona
    third   = (mcardona[iel,iion]/3) * (ennstar**3 - 343)  
    Elast   = chiK[iel,iion] - (((Zeff**2)*RydK)/(ennstar**2)) # \hat{E} from Cardona
    fine = gground[iel,iion] + (Gcardona[iel,iion]* np.exp(-Ecardona[iel,iion]/T))+(third*np.exp(-Elast/T)) # final partition eq. from Cardona
    
    if fine < gground[iel,iion]:
        fine = gground[iel,iion]
    if iion==iel+1:
        fine =  1.0
    if fine < 1.0:
        fine = 1.0
    # if fine < 1.: #1e-10: #1e-25, 1e-50: 
    #     return Udummy
    # else:
    return fine
Upart_fun=np.vectorize(Upart_fun)


def lambdae(Temp):
    denom1 = 2*pi*xmelectron*boltzk*Temp
    denom_fin = np.sqrt(denom1)
    frac = hPlanck/denom_fin
    
    return frac

## potential 3BR fix
# def tbr(iel,iion,CI,N,Temp):    
#     frac1 = 2/(N*(lambdae(Temp)**3)) # need n_e (electron density, same as N?)
#     top = Upart_fun(iel,iion+1,N,Temp)
#     bottom = Upart_fun(iel,iion,N,Temp)
#     frac2 = top/bottom
#     Si = frac1*frac2*np.exp(-chiK[iel,iion]/Temp)

#     if CI ==0: # this is a stop-gap for now, will need to revisit
#         BR=0
#     else:
#         BR = CI/Si # CI=collisional ionization component, calculated by either cfit or cheavy
#     return BR
# tbr=np.vectorize(tbr)

# def tbr(iel,iion,CI,N,Temp):    ##original
#     frac1 = 2/(N*(lambdae(Temp)**3)) # need n_e (electron density, same as N?)

#     top = Upart_fun(iel,iion+1,N,Temp)
#     bottom = Upart_fun(iel,iion,N,Temp)
#     frac2 = top/bottom

#     Si = frac1*frac2*np.exp(-chiK[iel,iion]/Temp)
    
#     BR = CI/Si # CI=collisional ionization component, calculated by either cfit or cheavy
    
#     return BR



## DEFINE GRAVITATIONAL RADIUS (in cm) ****************************************************************************************************************************
def Rg(Mass_bh): # Mass_bh=black hole mass in solar masses
    Mbh_grams = Mass_bh*xMsun # black hole mass in g
    rad_grav = (Gconst*Mbh_grams)/(clight**2) # Rg in cm
    
    return rad_grav 

## DEFINE LX ******************************************************************************************************************************************************
# effective hydrogen-ionizing luminosity
# integral of Lnu over all photon energies from 1 to 1000 Ryd
def LX_orig(freq_range,Lum_nu):
    ryd1_ind = np.argmin(np.abs(freq_range-ryd1_Hz))
    ryd1000_ind = np.argmin(np.abs(freq_range-ryd1000_Hz))

    x = freq_range[ryd1_ind:ryd1000_ind]
    y = Lum_nu[ryd1_ind:ryd1000_ind]
    
    integral = np.trapz(y,x) # use this for luminosity
    return integral

def LX(freq_range,J,radius): # J=Jnu
    ryd1_ind = np.argmin(np.abs(freq_range-ryd1_Hz))
    ryd1000_ind = np.argmin(np.abs(freq_range-ryd1000_Hz))

    x = freq_range[ryd1_ind:ryd1000_ind]
    y = J[ryd1_ind:ryd1000_ind]
    
    integral = np.trapz(y*16*(pi**2)*(radius**2),x) 
    return integral

## DEFINE WIND VELOCITY **********************************************************************************************************************************************
# function of radius
def u_r(radius,launch_radius,wind_vel):
    if radius<launch_radius:
        ur = 0
    if radius>=launch_radius:
        ur = wind_vel
    return ur
u_r=np.vectorize(u_r)

# DEFINE HYDROGEN NUMBER DENSITY AT LAUNCH RADIUS *********************************************************************************************************************
def nH(radius,launch_radius,n_launch,S,alpha):
    if radius<launch_radius:
        n_H = S*n_launch
    if radius>=launch_radius:
        n_H = n_launch*((launch_radius/radius)**alpha)
    return n_H

## DEFINE TXray *****************************************************************************************************************************************************
# photon energies from 100 to 10,000 eV for X-ray range
# def TXray2(freq_range,Lum_nu):
#     xray_ind1 = np.argmin(np.abs(freq_range-(100/hPlanck_eV))) # bottom of xray range
#     xray_ind2 = np.argmin(np.abs(freq_range-(1e5/hPlanck_eV))) # top of xray range
    
#     x_num = freq_range[xray_ind1:xray_ind2]
#     y_num = Lum_nu[xray_ind1:xray_ind2]*hPlanck*x_num
    
#     numerator = np.trapz(y_num,x_num) 

#     x = freq_range[xray_ind1:xray_ind2]
#     y = Lum_nu[xray_ind1:xray_ind2]
    
#     denom = np.trapz(y,x) 
    
#     xray_temp = numerator/denom/boltzk
#     return xray_temp

def TXray2(freq_range,J,radius):
    xray_ind1 = np.argmin(np.abs(freq_range-(100/hPlanck_eV))) # bottom of xray range
    xray_ind2 = np.argmin(np.abs(freq_range-(1e5/hPlanck_eV))) # top of xray range
    
    x_num = freq_range[xray_ind1:xray_ind2]
    y_num = J[xray_ind1:xray_ind2]*16*(pi**2)*(radius**2)*hPlanck*x_num
    
    numerator = np.trapz(y_num,x_num) 

    x = freq_range[xray_ind1:xray_ind2]
    y = J[xray_ind1:xray_ind2]*16*(pi**2)*(radius**2)
    
    denom = np.trapz(y,x) 
    
    xray_temp = numerator/denom/boltzk
    return xray_temp

def L_bol(freqs,Lum_nu):
    x = freqs 
    y = Lum_nu
    
    lbol = np.trapz(y,x) 
    
    return lbol


# def v_esc(radius,MBH_g):
#     esc = np.sqrt((2*Gconst*MBH_g)/radius)
#     return esc


## ORIGINAL PHOTOIONIZATION FUNCTIONS *********************************************************************************************************************************
def photocross(Z,iion,freq): 
    shell_inds = LUT[Z][iion]

    n95 = n95_tot[shell_inds]
    l95 = l95_tot[shell_inds]
    Eth95 = Eth95_tot[shell_inds]  # eV
    E095 = E095_tot[shell_inds]   #eV
    sig095 = sig095_tot[shell_inds] # Mb (1e-18 cm^2)
    ya95 = ya95_tot[shell_inds]
    P95 = P95_tot[shell_inds]
    yw95 = yw95_tot[shell_inds]

    E = hPlanck_eV*freq

    Q = 5.5+l95-(0.5*P95)
    Y = np.array([energy/E095 for energy in E])

    Fy = ((Y-1)**2+(yw95**2))*(Y**-Q)*(1+np.sqrt(Y/ya95))**(-P95)
    sigma_Mb = sig095*Fy # gives sigma_nu in Mb (lol Verner why u do dis)
    sigma_cm = sigma_Mb*1e-18

    for j in range(sigma_cm.shape[1]):
        for i in range(0,len(freq)):
            if E[i]<Eth95[j]:
                sigma_cm[i,j] = 0

    fin_sig_array = np.sum(sigma_cm,1) 
    return fin_sig_array

## ELEMENT ABUNDANCES AND MASSES **************************************************************************************************************************************
# in periodictable index[0] is the neutron
Ael = [el.mass for el in periodictable.elements] # atomic weights, recall: original fortran is 1-indexed
elname = [el.symbol for el in periodictable.elements] # element names, recall: original fortran is 1-indexed

Ael[0] = 0.0 # make neutron 0 bc we don't care about it
elname = elname[:nZmax+1] # truncate to match lengths
Ael = Ael[:nZmax+1] # truncate to match lengths

abunds = np.genfromtxt(abunds_file,delimiter='|',usecols=(0,1),comments='%') #column 1 is solar abundances
# abunds = np.genfromtxt(abunds_file,delimiter='|',usecols=(0,2),comments='%') #column 2 is 2*solar abundances
# abunds = np.genfromtxt(abunds_file,delimiter='|',usecols=(0,3),comments='%') #column 3 is 3*solar abundances
# abunds = np.genfromtxt(abunds_file,delimiter='|',usecols=(0,4),comments='%') #column 4 is 4*solar abundances

abun_by_n = abunds[:,1][:nZmax+1] # truncate to match lengths 

abun_by_mass = (abun_by_n * Ael)/np.sum(abun_by_n * Ael) # this is unitless
abun_by_mass[0] = -99.0 # dummy for the neutron

