import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from saturation_thermodynamics import get_saturation_thermodynamics
import constants as const
import xarray as xr

class WalkerCirculationModel:
    def __init__(self, y0=None, A=1e7, n=200, Cp=const.cp, mld=5, delta_p_T=800e2, 
                 c_q=0.009, 
                 r_hr=0.17, r_hs=0.17, r_l=6, 
                 tau_c=16*3600,
                 delta_S=80, delta_R=35, a_s=0.302, b_s=1,
                 a_LTS=0.381, a_hat=0.459, b_hat=0.316, M_sp=0.0427, M_qp=0.0507,
                 Ms0=3643, Mq0=3096, RH0=0.82, T0_ps=29, Ts0=30.2, 
                 P0=148, S_clr0=203, R_clr0=138, 
                 gamma_Ts_S=-6.28, gamma_q_S=1.64, gamma_T_S=1.51, 
                 gamma_Ts_R=-5.99, gamma_q_R=1.08, gamma_T_R=2.9):
    
        # Basic parameters
        self.y0 = y0 # Initial conditions
        self.n = n # Number of spatial points
        self.A = A # Domain width (m)
        self.C_o = mld*1e3*4184/Cp # Ocean mixed layer heat capacity (kg m-2), weird unit used to account for T being in energy units
        self.delta_p_T = delta_p_T # Tropospheric pressure depth (Pa)
        self.c_q = c_q # Bulk constant for evaporation (kg m^-2 s^-1)
        self.tau_c = tau_c # Convective adjustment timescale
        self.delta_S = delta_S # Change in S across domain (W m^-2)
        self.delta_R = delta_R # Constant atm heat export (W m^-2)
    
        # Parameters from QTCM v2.3
        self.a_LTS = a_LTS # LTS conversion factor
        self.a_hat = a_hat # Vertical average of a
        self.b_hat = b_hat # Vertical average of b
        self.a_s   = a_s # b(p_s)
        self.b_s   = b_s # a(p_s)
        self.M_sp = M_sp # dM_s/dT
        self.M_qp = M_qp # dM_q/dT
        
        # RCE parameters
        self.R_clr0 = R_clr0 # RCE clear-sky radiative flux divergence (W m^-2)
        self.S_clr0 = S_clr0 # RCE clear-sky surface radiative heating (W m^-2)
        self.RH_0 = RH0      # RCE surface relative humidity
        self.P_0 = P0        # RCE precipitation/evaporation rate (W m^-2)
        self.T_s0 = Ts0      # RCE ocean temperature (°C)
        self.T_0 = T0_ps     # RCE surface air temperature (°C)
        self.M_s0 = Ms0      # RCE gross dry stability (J kg^-1), default = 3643
        self.M_q0 = Mq0      # RCE gross moisture stratification (J kg^-1), default = 3096

        # Derived parameters
        self.D_q = -self.M_qp  # Horizontal moisture advection coefficient
        self.x = np.linspace(0,self.A, self.n) # gridpoints
        self.dx = self.A / (self.n - 1)  # Spatial step size
        
        # Clear-sky radiative flux perturbation parameters from QTCM v2.3
        # See end of Section 2 of PB2005 for unit conversion
        self.gamma_T_R = gamma_T_R/Cp  # dR_clr/dT (W m^-2 K^-1 --> W m^-2 J^-1 kg)
        self.gamma_q_R = gamma_q_R/Cp  # dR_clr/dq (W m^-2 K^-1 --> W m^-2 J^-1 kg)
        self.gamma_Ts_R = gamma_Ts_R/Cp  # dR_clr/dT_s (W m^-2 K^-1 --> W m^-2 J^-1 kg)
        self.gamma_T_S = gamma_T_S/Cp  # dS_clr/dT (W m^-2 K^-1 --> W m^-2 J^-1 kg)
        self.gamma_q_S = gamma_q_S/Cp  # dS_clr/dq (W m^-2 K^-1 --> W m^-2 J^-1 kg)
        self.gamma_Ts_S = gamma_Ts_S/Cp  # dS_clr/dT_s (W m^-2 K^-1 --> W m^-2 J^-1 kg)

        # Cloud feedback parameters
        self.r_hr = r_hr # High-cloud atm radiative heating feedback, dimensionless, default = 0.17
        self.r_hs = r_hs # High-cloud surface radiative heating feedback, dimensionless, default = 0.17
        self.r_l = r_l/Cp   # Low-cloud radiative heating feedback (W m^-2 K^-1 --> W m^-2 J^-1 kg)

        if self.y0==None:
            # Set initial conditions
            T0 = np.zeros(self.n) 
            q0 = np.zeros(self.n) 
            Ts0 = np.zeros(self.n) 
            self.y0 = np.concatenate([T0, q0, Ts0])

    
    def M_s(self, T, q):
        return self.M_s0 + self.M_sp *T #* np.maximum(T,q) (Alternative from sobel's review paper)


    def M_q(self, q): 
        return self.M_q0 + self.M_qp * q

    
    def evaporation(self, T, T_s, q, P, u):
        # P and u are extra variables here, included for when I was messing with the Evap formulation
        """
        Comparing my cloud-free solutions with Fig. 3 of PB05 reveals some subtle differences. 
        I haven't tracked them down yet, but I note that the solution is quite sensitive to
        precise details of the evaporation parameterization. 
        """
        q_star_s = self.saturation_specific_humidity(self.T_s0 + T_s/const.cp +273.15)
        q_0 = self.RH_0 * self.saturation_specific_humidity(self.T_0+273.15) 
        return self.c_q * (q_star_s*const.Lv - q_0*const.Lv - self.b_s * q) 


    def saturation_specific_humidity(self, T):
        # results indistinguishable if I use metpy's implementation
        [es,qs,rs,L] = get_saturation_thermodynamics(T,1e5,thermo_type='simple')
        return qs # units: kg/kg 


    def precipitation(self, q, T):
        return np.maximum((self.delta_p_T / 9.81) * (q - T) / self.tau_c + self.P_0, 0)


    def F_lowcloud(self, T, T_s, P):
        a_lts = 0.381
        LTS = (a_lts*T-T_s)
        return (1-np.heaviside(P, 0))*self.r_l*LTS
        
    
    def R(self, T, q, T_s, P):
        R_clr = self.R_clr0 + self.gamma_T_R * T + self.gamma_q_R * q + self.gamma_Ts_R * T_s
        R_cld = -self.r_hr * P  + 0.5 * self.F_lowcloud(T, T_s, P) # High cloud effect + Low cloud effect
        return R_clr + R_cld + self.delta_R


    def S(self, T, q, T_s, P, x):
        S_clr = self.S_clr0 + self.gamma_T_S * T + self.gamma_q_S * q + self.gamma_Ts_S * T_s
        S_cld = -self.r_hs * P  - 0.5 * self.F_lowcloud(T, T_s, P)  # High cloud effect + Low cloud effect
        S_ocn = 10 - self.delta_S * x / self.A
        return S_clr + S_cld + S_ocn


    def get_dTdt(self, P, R, x):
        return np.trapezoid(P-R, x) * 9.81 / (self.A * self.delta_p_T * self.a_hat)


    def get_dSSTdt(self, T, T_s, q, P, u):
        E = self.evaporation(T, T_s, q, P, u)
        S = self.S(T, q, T_s, P, self.x)
        return (S - E) / self.C_o

    
    def get_dqdt(self, T, T_s, q, P, omega, u):
        E = self.evaporation(T, T_s, q, P, u)
        M_q = self.M_q(q)
        dqdx = np.gradient(q, self.dx)
        return ( 9.81 / self.delta_p_T / self.b_hat ) * ( (E - P) + (M_q * (omega / 9.81)) - (self.delta_p_T / 9.81) * self.D_q * u * dqdx )

    
    def get_omega(self, P, R, T, x, q):
        M_s = self.M_s(T, q)
        dTdt = self.get_dTdt(P, R, x)
        return (P - R - self.a_hat * (self.delta_p_T / 9.81) * dTdt) * (9.81 / M_s) # Gives units of Pa/s, as [T]=J/kg

    
    def model_equations(self, t, y):
        
        T, q, T_s = y[:self.n], y[self.n:2*self.n], y[2*self.n:]

        # DEBUGGING OPTION, OVERRIDE SST EQUATION
        #T_s = 3*np.cos(np.linspace(-np.pi, np.pi,n))+27

        # Equation 10: Closed box constraint
        P = self.precipitation(q, T)
        R = self.R(T, q, T_s, P)
        dTdt = self.get_dTdt(P, R, self.x) 

        # Equation 6: Solve for omega (vertical velocity)
        omega = self.get_omega(P, R, T, self.x, q) 
                
        # Equation 9: Solve for u (horizontal velocity)
        # cumulatively integrate, then add a zero at the beginning
        u = np.insert(cumulative_trapezoid(omega/self.delta_p_T, self.x), 0, 0)
        u[-1] = 0 # u=0 at edges of domain, see PB05's appendix
        
        # Equation 7: Moisture equation
        dqdt = self.get_dqdt(T, T_s, q, P, omega, u)
        
        # Equation 8: Surface temperature equation
        dSSTdt = self.get_dSSTdt(T, T_s, q, P, u)

        return np.concatenate([np.full(self.n, dTdt), dqdt, dSSTdt])
            
            
    def solve(self, t_span, t_eval=None):
        sol = solve_ivp(self.model_equations, t_span, self.y0, t_eval=t_eval)
        return sol
