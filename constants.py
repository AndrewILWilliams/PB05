import numpy as np


cday    = 86400.0         # sec in calendar day ~ sec
cyear   = cday*365.25     # sec in calendar year ~ sec
sday    = 86164.0         # sec in siderial day ~ sec
omega   = 2*np.pi/sday # earth rot ~ rad/sec
a       = 6.37122e6       # radius of earth ~ m
g       = 9.80616         # acceleration of gravity ~ m/s^2


sigma   = 5.67e-8         # Stefan-Boltzmann constant ~ W/m^2/K^4
boltz   = 1.38065e-23     # Boltzmann's constant ~ J/K/molecule
avogad  = 6.02214e26      # Avogadro's number ~ molecules/kmole
c0      = 2.99792458E8    # Speed of light in a vacuum (m/s)
planck  = 6.6260755E-34   # Planck's constant (J.s) 
rgas    = avogad * boltz  # Universal gas constant ~ J/K/kmole
mwdair  = 28.966          # molecular weight dry air ~ kg/kmole
mwwv    = 18.016          # molecular weight water vapor
rd      = rgas/mwdair     # Dry air gas constant     ~ J/K/kg
rv     = rgas/mwwv        # Water vapor gas constant ~ J/K/kg
zvir    = (rv/rd)-1.      # RWV/RDAIR - 1.0
pstd    = 101325.0        # standard pressure ~ pascals


tktrip  = 273.16          # triple point of water        ~ K
tkfrz   = 273.15          # freezing T of water          ~ K
rhodair = pstd/(rd*tkfrz) # density of dry air at STP  ~ kg/m^3
rhow    = 1.000e3         # density of water     ~ kg/m^3
rhosw   = 1.026e3         # density of sea water ~ kg/m^3
cp      = 1.00464e3       # specific heat of dry air   ~ J/kg/K
cpv     = 1.810e3          # specific heat of water vap ~ J/kg/K
cpvir   = (cpv/cp)-1.     # CPWV/CPDAIR - 1.0
cpw     = 4.188e3         # specific heat of h2o ~ J/kg/K
cpi     = 2.11727e3       # specific heat of ice ~ J/kg/K
cpsw    = 3.996e3         # specific heat of sea water ~ J/kg/K
Lf      = 3.337e5         # latent heat of fusion      ~ J/kg
#Lv      = 2.501e6         # latent heat of evaporation ~ J/kg
Lv      = 2.50e6         # latent heat of evaporation ~ J/kg
Ls      = Lf + Lv         # latent heat of sublimation ~ J/kg




mb_to_Pa = 100.           # conversion factor from mb to Pa


