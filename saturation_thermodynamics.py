import numpy as np
import constants as const


def get_saturation_thermodynamics(T,p,thermo_type='simple'):
    #thermo_type = 'simple' or 'era' are implemented


    Rd = const.rd
    Rv = const.rv
    gc_ratio = Rd/Rv


    if thermo_type == 'era':
        # ECMWF formulation described by Simmons et al. (1999: QJRMS, 125,
        # 353--386), which uses saturation over ice for temperatures
        # less than 250 K and a quadratic interpolation between
        # saturation over ice and over liquid water for temperatures
        # between 250 K and 273 K
    
        # coefficients in Tetens saturation vapor pressure es = es0 * exp(a3 * (T-T0)/(T-a4))      
        es0       = 611.21  # saturation vapor pressure at T0 (Pa)
        T0        = 273.16  # (K)
        Ti        = T0 - 23 # (K)


        a3l       = 17.502  # liquid water (Buck 1981)
        a4l       = 32.19   # (K)
  
        a3i       = 22.587  # ice (Alduchov and Eskridge 1996)
        a4i       = -0.7    # (K)
  
        # saturation vapor pressure over liquid and ice
        esl       = es0 * np.exp(a3l * (T - T0)/(T - a4l))
        esi       = es0 * np.exp(a3i * (T - T0)/(T - a4i))
        # latent heat of sublimation 
        Ls0       = 2.834e6 # latent heat of sublimation  (J / kg) [+- 0.01 error for 173 K < T < 273 K]


        # set up output arrays
        Ls = Ls0 * np.ones(T.shape)


        # latent heat of vaporization
        Lv0       = 2.501e6 # latent heat of vaporization at triple point (J / kg)
        cpl       = 4190    # heat capacity of liquid water (J / kg / K)
        cpv       = 1870    #heat capacity of water vapor (J / kg / K)      
        Lv        = Lv0 - (cpl - cpv) * (T - T0)
  
        # compute saturation vapor pressure and latent heat over liquid/ice mixture


        #iice      = T <= Ti
        #iliquid   = T >= T0
        #imixed    = (T > Ti) & (T < T0)
        iice = np.where(T <= Ti)
        iliquid = np.where(T >= T0)
        imixed = np.where((T > Ti) & (T < T0))


        #print(iice)
        #print(iliquid)
        #print(imixed)


        #print(len(iice))
        #print(len(iliquid))
        #print(len(imixed))
        
        es        = np.nan * np.ones(T.shape)
        L         = np.nan * np.ones(T.shape)
        a         = np.nan * np.ones(T.shape)  


        es = esl
        L = Lv


        es = np.where(T <= Ti, esi, es)
        L = np.where(T <= Ti, Ls, Lv)


        a = np.where((T > Ti) & (T < T0), ( (T - Ti)/(T0 - Ti) )**2, a)
        es = np.where((T > Ti) & (T < T0), (1-a) * esi + a * esl, es)
        L = np.where((T > Ti) & (T < T0), (1-a) * Ls + a * Lv, L)


    elif thermo_type == 'simple':
        # simple formulation
        # assuming constant latent heat of vaporization and only one
        # vapor-liquid phase transition
        # used in O'Gorman & Schneider 2008 iGCM


        T0        = 273.16 # K
        es0       = 610.78 # Pa
        L         = const.Lv 
    
        # saturation vapor pressure
        es        = es0 * np.exp(L/Rv*(1.0/T0 - 1.0/T));
        
    else:
        raise ValueError("Unknown type of computing saturation quantities.")




    # saturation mixing ratio
    rs          = gc_ratio * es / (p - es);
    
    # saturation specific humidity 
    qs          = rs / (1 + rs); 


    return es, qs, rs, L
