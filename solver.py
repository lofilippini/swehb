import numpy as np
import time
import warnings
import matplotlib.pyplot as plt

from bathymetry import build_bathymetry

class SWEHBSolver:
    def __init__(self, params, surge = False, reservoir = False, rollwaves = False):
        """
        Initialize the SWE HB Solver with the given parameters.
        params: dictionary containing:
            h0, ty, kn, m, rho (fluid properties)
            g (gravity)
            theta, M (geometry parameters)
            Nx, CFL, tend (numerical parameters)
            xL, xR (domain limits)
        """
        self._validate_params(params, surge, reservoir, rollwaves)

        self.surge = surge
        self.reservoir = reservoir
        self.roll = rollwaves

        if self.roll == True: self.surge = self.reservoir = False

        self.g = 9.81
        self.h0 = params['h0']
        self.ty = params['ty']
        self.kn = params['kn']
        self.m = params['m']
        self.rho = params['rho']
        self.theta = params['theta']
        
        self.Nx = params['Nx']
        self.CFL = params['CFL']
        self.tend = params['tend']
        
        self.M = params['M']
        self.xL = params['xL']
        self.xR = params['xR']
        
        # Derived parameters
        self.Lambda0 = np.sin(self.theta)
        self.Xi0 = self.ty / (self.rho * self.g * self.h0 * self.Lambda0)
        self.hcrit = self.ty / (self.rho * self.g * self.Lambda0)
        
        term1 = (self.rho * self.g * self.Lambda0 * ((self.h0 * (1 - self.Xi0))**(self.m + 1)) / self.kn)**(1/self.m)
        term2 = (1 - self.m / (2 * self.m + 1) * (1 - self.Xi0))
        self.u0 = (self.m / (self.m + 1)) * term1 * term2
        self.umax = self.u0*(2*self.m + 1)/(1 + self.m + self.m*self.Xi0)
        
        self.Fr0 = self.u0/(np.sqrt(self.g * np.cos(self.theta) * (self.h0**0.5)))

        self.mu0 = self.kn*self.mu(self.Xi0)*(self.u0/self.h0)**self.m
        self.tb0 = self.ty + self.mu0
        self.z0 = self.h0 * (1 - self.Xi0)
        self.ss = self.u0/self.h0
        self.q0 = self.h0 * self.u0

        # Roll wave boundary condition parameters (see set_roll_wave to customize)
        self.roll_amp = 0.02
        self.roll_freq = 1.5
        self.roll_u0 = self.u0
        
        # Grid setup
        self.dx = (self.xR - self.xL) / self.Nx
        self.x = np.linspace(self.xL, self.xR, self.Nx)
        self.xb = np.linspace(self.xL + 0.5 * self.dx, self.xR - 0.5 * self.dx, self.Nx)
        
        # Bathymetry setup
        self.bathb = np.zeros(self.Nx, dtype=np.float128)    
        self.u = self.u0 * np.ones(self.Nx + 1)
        self.ub = 0.5 * (self.u[0:self.Nx] + self.u[1:self.Nx+1])
        
        self.dt = 0.0 # Will be set in solve loop
        self.tic = 0.0 # Time counter for oscillating boundary conditions
 
        self.center = 0
        self.norm = self.M
       
        self.epsilon = self.h0/(self.xR - self.xL)

        # Bathymetry bookkeeping (populated by set_bathymetry)
        self.bath_type = None
        self.x_left = None
        self.x_right = None
        self.x_flat_left = None
        self.x_flat_right = None


    @staticmethod
    def _validate_params(params, surge, reservoir, roll):
        """Validates `params` and case flags, raising on invalid setups and warning on risky ones."""
        required_keys = ['h0', 'ty', 'kn', 'm', 'rho', 'theta', 'Nx', 'CFL', 'tend', 'M', 'xL', 'xR']
        missing = [k for k in required_keys if k not in params]
        if missing:
            raise ValueError(f"Missing required parameter(s) in `params`: {missing}")

        if params['h0'] <= 0:
            raise ValueError("h0 (reference depth) must be positive")
        if params['ty'] < 0:
            raise ValueError("ty (yield stress) must be non-negative")
        if params['kn'] <= 0:
            raise ValueError("kn (consistency index) must be positive")
        if params['m'] <= 0:
            raise ValueError("m (flow index) must be positive")
        if params['rho'] <= 0:
            raise ValueError("rho (density) must be positive")
        if params['Nx'] <= 0:
            raise ValueError("Nx (number of cells) must be positive")
        if params['CFL'] <= 0:
            raise ValueError("CFL must be positive")
        if params['tend'] <= 0:
            raise ValueError("tend (simulation end time) must be positive")
        if params['xL'] >= params['xR']:
            raise ValueError("xL must be smaller than xR")

        if params['CFL'] > 1:
            warnings.warn(f"CFL = {params['CFL']} > 1 may lead to an unstable simulation")
        if not (0 < params['theta'] < np.pi/2):
            warnings.warn(f"theta = {params['theta']:.3f} rad is outside the usual (0, pi/2) slope range")
        if params['Nx'] < 50:
            warnings.warn(f"Nx = {params['Nx']} is quite coarse and may not resolve the flow well")
        if params['M'] <= 0:
            warnings.warn("M (obstacle/normalization scale) should be positive")

        if reservoir and not surge:
            warnings.warn("reservoir=True has no effect unless surge=True (reservoir initial condition is only built in the surge case)")
        if roll and surge:
            warnings.warn("rollwaves and surge are typically mutually exclusive flow cases; combining them may produce unexpected results")


    def set_bathymetry(self, bath_type, center=0, contour_file=None, a=1):
        """
        Builds one of the bathymetry profiles defined in bathymetry.py and stores it
        in self.bathb, along with the associated flat/slope region markers
        (x_left, x_right, x_flat_left, x_flat_right) used by the solver's boundary handling.

        bath_type: one of 'dead_zones', 'rectangle', 'squared_trapezoid',
            'semi_circular', 'bump', 'ramp', 'flat'/'none', 'sinusoidal',
            'contour' (loads bathymetry from contour_file), 'custom' (edit
            custom_bathymetry() in bathymetry.py)
        center: horizontal offset of the obstacle (where applicable)
        a: shape exponent used by the 'bump' profile
        """
        self.bath_type = bath_type
        self.bathb, self.x_left, self.x_right, self.x_flat_left, self.x_flat_right = build_bathymetry(
            self.xb, self.M, bath_type, center=center, a=a, contour_file=contour_file
        )

        return self.bathb


    def set_roll_wave(self, amp=None, freq=None, u0=None):
        """
        Configures the oscillating inflow boundary condition used when roll=True.
        amp: relative amplitude of the velocity perturbation (default 0.02)
        freq: perturbation frequency (default 1.5)
        u0: base inflow velocity to perturb (defaults to the solver's steady u0)
        """
        if amp is not None:
            self.roll_amp = amp
        if freq is not None:
            self.roll_freq = freq
        if u0 is not None:
            self.roll_u0 = u0


    def check_inlet(self):
        h0 = self.h0
        u0 = self.u0
        print("--------------------------------")
        print("Check inlet:")
        print(f"theta = {np.degrees(self.theta):.2f} degrees\nh0 = {self.h0:.3f} m\nu0 = {self.u0:.3f} m/s\numax = {self.umax:.3f} m/s")
        print(f"mu0 = {self.mu0:.3f} Pa.s\ntb0 = {self.tb0:.3f} Pa\nq0 = {self.q0:.5f} m^2/s")
        print(f"\nInlet condition dimensionless numbers:\n"
              f"Fr = {self.Froude(u0, h0):.4f}\n"
              f"Re = {self.Reynolds(u0, h0):.4f}\n"
              f"Bi = {self.Bingham(u0, h0):.4f}\n"
              f"Pl = {self.Plastic(u0, h0):.4f}\n"
              f"Xi0 = {self.Xi(h0, self.Lambda0)[0]:.4f}\n")
              
        print("--------------------------------\n")

        if self.roll:
            print(f"ATENTION: Roll wave boundary condition is enabled!")


    def check_case(self, zeta_file=None, u_file=None, show=True):
        """
        Builds the initial condition from the current bathymetry (self.bathb) and
        plots the bathymetry, initial free surface and initial velocity so the
        user can visually validate the case before running the solver.
        """
        self.set_case(self.bathb, self.x_left, self.x_right, self.x_flat_left, self.x_flat_right,
                      zeta_file=zeta_file, u_file=u_file)

        self.check_inlet()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

        ax1.plot(self.xb, self.bathb, color='saddlebrown', label='Bathymetry')
        ax1.plot(self.xb, self.zetab, color='royalblue', label='Free surface (zeta)')
        # ax1.fill_between(self.xb, self.bathb, self.zetab, color='royalblue', alpha=0.2)
        ax1.set_ylabel('Elevation')
        ax1.set_title(f"Case check - bathymetry: '{self.bath_type}'")
        if self.roll:
            ax1.set_title(f"Case check - bathymetry: '{self.bath_type}' \n ATENTION: Roll wave boundary condition is enabled!")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.xb, self.hb, color='k', label='Initial depth (h)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Depth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(self.xb, self.ub, color='k', label='Initial velocity (u)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('Velocity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()

        return fig, (ax1, ax2)


    def set_case(self, bathymetry, x_left=None, x_right=None, x_flat_left=None, x_flat_right=None, zeta_file = None, u_file = None):
        """
        Sets the case bathymetry and initial conditions according to the case's geometry and type. 
        """
        self.bathb = bathymetry.copy()

        global l

        def reservoirfunc(self, x, h_g):
            global l
            h = np.zeros_like(x)
            mask = (x >= 0) & (x <= l)
            h[mask ] = h_g + (x[mask] - l) * np.tan(self.theta)
            return h

        if self.surge:
            self.u = np.zeros(self.Nx + 1)
            self.bathb = bathymetry.copy()
            
            if self.reservoir:
                l = 0.50
                h_g = self.h0
                self.bathb[self.xb <= 0] = h_g
                self.zetab = reservoirfunc(self, self.xb, h_g) + self.bathb

                self.u0 = 0
                
            else:
                self.zetab = self.bathb + 1e-12
                self.zetab[0] = self.h0
            
        else:
            self.zetab = (np.ones(self.Nx, dtype=np.float128)*(self.h0) + self.bathb)
            self.u = self.u0 * np.ones(self.Nx + 1)


        if zeta_file is not None:
            # Assumes numpy is available, but keeps dependency minimal
            zeta_data = np.loadtxt(zeta_file, delimiter=',', skiprows=1)
            zeta_data = np.interp(self.xb, zeta_data[:, 0]*self.norm, zeta_data[:, 1]*self.norm)
            self.zetab = zeta_data

        if u_file is not None:
            xu = np.linspace(self.xL, self.xR, self.Nx + 1)
            u_data = np.loadtxt(u_file, delimiter=',', skiprows=1)
            u_data = np.interp(xu, u_data[:, 0]*self.norm, u_data[:, 1]*self.norm)
            self.u = u_data


        self.hb = np.maximum(self.zetab - self.bathb, 1e-12)        
        self.ub = 0.5 * (self.u[0:self.Nx] + self.u[1:self.Nx+1])

        self.x_left = x_left
        self.x_right = x_right
        self.x_flat_left = x_flat_left
        self.x_flat_right = x_flat_right


    # Static helper methods
    def Froude(self, u, h):
        """Calculate local Froude number."""
        h_safe = np.maximum(h, 1e-12)
        return u / (np.sqrt(self.g * np.cos(self.theta)) * (np.abs(h_safe) ** 0.5))

    def Bingham(self,u, h):
        """Calculate local Bingham/Herschel-Bulkley number."""
        h_safe = np.maximum(h, 1e-12)
        ss = u / h_safe
        return self.ty / (self.kn * (np.abs(ss)**self.m))

    def Reynolds(self, u, h):
        """Calculate local Reynolds number."""
        h_safe = np.maximum(h, 1e-12)
        ss = u / h_safe
        return (self.rho * (u**(2))) / (self.ty + self.kn * (np.abs(ss)**self.m))
    

    def Plastic(self, u, h):
        """Calculate local Plasticity number."""
        h_safe = np.maximum(h, 1e-12)
        ss = u / h_safe
        Pl = self.ty / (self.ty + self.kn * (np.abs(ss)**self.m))
        Pl = np.where(Pl >= 1, 1, Pl)
        return Pl
    

    def Xi(self, h, Lbd, u = None):
        """Calculate local Xi parameter."""
        h_safe = np.maximum(h, 1e-12)
        Xi = self.ty / (self.rho * self.g * h_safe * np.abs(Lbd))
        Xi_filter = np.abs(Xi) >= 1

        if isinstance(h, np.ndarray) == False:
            return Xi, Xi_filter
        else:
            pass
        
        # Xi filter
        if (self.reservoir == False) and (h_safe[-1] > 1e-6):      
            Xi = np.where(Xi >= 1, self.Xi0, Xi)
            if u is not None:
                if u.size != h.size:
                    u = 0.5*(u[0:self.Nx] + u[1:self.Nx+1])
                Bi = self.Bingham(u, h)
                Fr = self.Froude(u, h)
                Xi = np.where((Bi < 1) & (Fr > 1), 0, Xi)  
                return Xi, Xi_filter 
            else:
                return Xi, Xi_filter
        
        else:
            Xi = np.where(Xi >= 1, 0.99, Xi)
            return Xi, Xi_filter


    def Lambda(self, f, x):
        grad_f = np.gradient(f, x)
        l = np.sin(self.theta) - grad_f * np.cos(self.theta) 
    
        return l


    def mu(self, Xi):
        """Calculate effective viscosity multiplier."""

        Xi = np.minimum(Xi, 0.99)  # or 1 - 1e-6
        Xi_alt = ((1 + 2*self.m)/(1 + self.m*(1 + Xi))*(1 + self.m)/(self.m*(1 - Xi)))
        Xi_alt = np.maximum(Xi_alt, 0)  
        muv = Xi_alt**self.m

        return muv


    def tau(self, Xi, u, h):
        """Calculate basal shear stress."""
        h_safe = np.maximum(h, 1e-12)
        ss = u / h_safe
        t = (self.ty + self.kn * self.mu(Xi) * ((np.abs(ss))**self.m))
        return t


    def get_dimensionless_numbers(self):
        Fr = self.Froude(self.ub, self.hb)
        Re = self.Reynolds(self.ub, self.hb)
        Bi = self.Bingham(self.ub, self.hb)
        Pl = self.Plastic(self.ub, self.hb)
        return Fr, Re, Bi, Pl


    def _LinearPartCoeff(self, zetab, hb, u):
        Hxm = np.zeros(self.Nx)
        Hxp = np.zeros(self.Nx)
        Bxm = np.zeros(self.Nx)
        Bxp = np.zeros(self.Nx)
        xm = np.zeros(self.Nx)
        xp = np.zeros(self.Nx)
        Xim = np.zeros(self.Nx, dtype=np.float128)
        Xip = np.zeros(self.Nx, dtype=np.float128)
        mum = np.zeros(self.Nx, dtype=np.float128)
        mup = np.zeros(self.Nx, dtype=np.float128)

        Hxm[0          ] = hb[0]
        Hxm[1:self.Nx  ] = 0.5*(hb[1:self.Nx] + hb[0:self.Nx-1])
        Hxp[0:self.Nx-1] = Hxm[1:self.Nx]
        Hxp[self.Nx - 1] = hb[self.Nx - 1]

        Bxm[0          ] = self.bathb[0]
        Bxm[1:self.Nx  ] = 0.5*(self.bathb[1:self.Nx] + self.bathb[0:self.Nx-1])
        Bxp[0:self.Nx-1] = Bxm[1:self.Nx]
        Bxp[self.Nx - 1] = self.bathb[self.Nx - 1]

        xm[0          ] = self.x[0]
        xm[1:self.Nx  ] = 0.5*(self.x[1:self.Nx] + self.x[0:self.Nx-1])
        xp[0:self.Nx-1] = xm[1:self.Nx]
        xp[self.Nx - 1] = self.x[self.Nx - 1]

        if self.grad == True:
            Lambda_p = self.Lambda(Bxp + Hxp, xp)
            Lambda_m = self.Lambda(Bxm + Hxm, xm)
        else:
            Lambda_p = self.Lambda(Bxp, xp)
            Lambda_m = self.Lambda(Bxm, xm)

        Xim,_ = self.Xi(Hxm, Lambda_m, u[0:self.Nx])
        Xip,_ = self.Xi(Hxp, Lambda_p, u[1:self.Nx+1])

        mup = self.kn * self.mu(Xip)
        mum = self.kn * self.mu(Xim)
        ssp = u[1:self.Nx+1]/Hxp
        ssm = u[0:self.Nx]/Hxm

        taub_p = 1/self.rho*(self.ty + mup *((np.abs(ssp))**self.m))
        taub_m = 1/self.rho*(self.ty + mum * ((np.abs(ssm))**self.m))
        
        rhs = hb - \
            self.dt/self.dx * ( \
              Hxp * u[1:self.Nx+1] + self.dt*(-taub_p + self.g*np.sin(self.theta)*Hxp) + \
            - Hxm * u[0:self.Nx  ] - self.dt*(-taub_m + self.g*np.sin(self.theta)*Hxm)
            )
        
        return Hxm, rhs


    def ExactRiemannProblem(self, qL, qR):
        # qL, qR are interface states with same shape
        # Shock case: qL > qR, 
        # speed s = (fR-fL)/(qR-qL) = 0.5*(qL+qR) ==> BURGERS EQUATION
        shock = qL > qR # boolean array indicating shock cases
        s = 0.5 * (qL + qR) # hankine-hugoniot condition

        # Rarefaction case:
        # if qL > 0 -> q = qL
        # elif qR < 0 -> q = qR
        # else -> q = 0 (because xi = 0)
        q_raref = np.where(qL > 0.0, qL, np.where(qR < 0.0, qR, 0.0))

        # Shock selection with xi=0:
        # 0 <= s -> q = qL, else q = qR
        q_shock = np.where(s >= 0.0, qL, qR)

        q_star = np.where(shock, q_shock, q_raref)
        # qstar uses the boolean shock array to identify where there are
        # shock waves (1) and where there are rarefaction waves (0)
        
        return 0.5 * q_star * q_star

    

    def _MomentumConvection(self, u):
        ustar = u.copy()
        dtdx = self.dt / self.dx
        self.tic += self.dt

        f = lambda q : 0.5 * q**2

        # Set boundary conditions
        if self.reservoir: # Reservoir/dam-break boundary condition
            ustar[0] = ustar[0]
            
        elif self.roll: # Roll wave boundary condition (see set_roll_wave)
            ustar[0] = self.roll_u0*(1 + self.roll_amp*np.sin(2 * np.pi * self.tic * self.roll_freq))
        else: # Constant discharge boundary condition
            ustar[0] = self.u0

        # Godunov flux scheme
        qL = u[0:self.Nx]
        qR = u[1:self.Nx+1]
        F = self.ExactRiemannProblem(qL, qR)

        # Cell update i = 1..Nx-1: uses F_{i+1/2} - F_{i-1/2}
        ustar[1:self.Nx] -= dtdx * (F[1:self.Nx] - F[0:self.Nx-1])
        
        return ustar


    def _MatVectProd_zeta(self, zetab, Hxm):
        fm = np.zeros(self.Nx)
        fp = np.zeros(self.Nx)

        fm[0          ] = 0 
        fm[1:self.Nx  ] = self.g * self.dt / self.dx * Hxm[1:self.Nx] * np.cos(self.theta) * (zetab[1:self.Nx] - zetab[0:self.Nx-1])
        fp[0:self.Nx-1] = fm[1:self.Nx]
        fp[self.Nx-1  ] = 0

        Mzeta = - self.dt/self.dx * (fp - fm)
        return Mzeta


    def _MatVectProdNewton(self, zetab, Hxm, wet):
        Mzeta = zetab*wet + self._MatVectProd_zeta(zetab, Hxm)
        return Mzeta


    def _CGsolver(self, rhs, Hxm, wet):
        tol = 1e-12
        N = rhs.size
        x = rhs.copy()
        r = rhs - self._MatVectProdNewton(x, Hxm, wet)
        p = r.copy()
        err = np.sum(r * r)
        
        for k in range(N):
            if err < tol:
                return x
            Ap = self._MatVectProdNewton(p, Hxm, wet)
            alpha = err / np.sum(p * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            err_new = np.sum(r * r)
            p = r + (err_new / err) * p
            err = err_new

        print(f"CG does NOT converge, residual = {err}")
        return x


    def _VelocityUpdate(self, u, zetab, hb):
        u = u.copy()

        zeta_itf = np.zeros(self.Nx+1)
        h_itf = np.zeros(self.Nx+1)
        b_itf = np.zeros(self.Nx+1)
        x = np.linspace(self.xL, self.xR, self.Nx+1)

        zeta_itf[0        ] = zetab[0]
        zeta_itf[1:self.Nx] =  0.5*(zetab[1:self.Nx] + zetab[0:self.Nx-1])
        zeta_itf[self.Nx  ] = zetab[self.Nx - 1]

        h_itf[0        ] = hb[0]
        h_itf[1:self.Nx] =  0.5*(hb[1:self.Nx] + hb[0:self.Nx-1])
        h_itf[self.Nx  ] = hb[self.Nx-1]

        b_itf[0        ] = self.bathb[0]
        b_itf[1:self.Nx] = 0.5*(self.bathb[1:self.Nx] + self.bathb[0:self.Nx-1])
        b_itf[self.Nx  ] = self.bathb[self.Nx-1]
        
        if self.grad == True:
            Lambdab = self.Lambda(zeta_itf, x)
        else:
            Lambdab = self.Lambda(b_itf, x)

        Xip, plastic = self.Xi(h_itf, Lambdab, u)
        Pl = self.Plastic(u, h_itf)

        mu = self.kn*self.mu(Xip)
        ss_alt = u/h_itf

        #taub = self.tau(Xip, u, hp)/self.rho
        taub_itf = 1/self.rho*(self.ty + mu*((np.abs(ss_alt))**self.m))

        u[1:self.Nx] = u[1:self.Nx] \
            - self.g*self.dt/self.dx*np.cos(self.theta)*(zetab[1:self.Nx] - zetab[0:self.Nx-1]) \
            + self.dt*self.g*np.sin(self.theta)\
            - self.dt/h_itf[1:self.Nx]*taub_itf[1:self.Nx]
        
        u[self.Nx] = u[self.Nx - 1]  
        
        u = np.where(u < 1e-9, 0, u)
        

        return u
    

    def _initialize_plots(self):
        # Setup plotting
        plt.close('all')
        fig, axs = plt.subplots(2, 3, figsize=(18, 8), dpi=100)
        fig_dom, axs_dom =  plt.subplots(3, 1, figsize=(12, 9), dpi=100)        
        fig_dom.suptitle(f'Domain Overview', fontsize=10)
        ax_dom = axs_dom[0]
        bath_dom, = ax_dom.plot(self.xb/self.norm, self.bathb/self.norm, 'k-', lw=1, label='Bathymetry')
        zeta_dom, = ax_dom.plot(self.xb/self.norm, self.zetab/self.norm, 'r-', lw=1, label='Free Surface')
        plug_dom, = ax_dom.plot(self.xb/self.norm, (self.zetab - self.h0*self.Xi0)/self.norm, 'b-', lw=1, label='Shear Surface')

        # ref_data = np.loadtxt('/home/oznerol/swehb/boghiFr175alpha06.csv', delimiter=',')
        # x_ref = ref_data[:, 0]
        # z_ref = ref_data[:, 1]
        # ax_dom.plot(x_ref, z_ref, 'g-', lw=1, label='Boghi et al. (reference)')

        # ref_data = np.loadtxt('/home/oznerol/swehb/scripts/contours/H50/freesurface_H50.csv', delimiter=',', skiprows=1)
        # x_ref = ref_data[:, 0]-30
        # z_ref = ref_data[:, 1]
        # ax_dom.plot(x_ref, z_ref, 'g-', lw=1, label='OpenFOAM')


        ax_dom.set_title('Domain Overview', fontsize=10)
        ax_dom.set_xlabel('x/h0')
        ax_dom.set_ylabel('z/h0')
        # ax_dom.set_xlim(-25, 25)
        # ax_dom.set_ylim(0, 6)
        ax_dom.legend(loc='upper right', fontsize='small')
        ax_dom.grid(True, linestyle='--')

        # Top Left: Surface
        ax_surf = axs[0, 0]
        line_bath, = ax_surf.plot(self.xb/self.norm, self.bathb/self.norm, 'k-', lw=1, label='Bathymetry')
        line_zeta, = ax_surf.plot(self.xb/self.norm, self.zetab/self.norm, 'r-', lw=1)#, label='Free Surface')
        line_plug, = ax_surf.plot(self.xb/self.norm, self.zetab/self.norm - self.h0*self.Xi0/self.norm, 'b-', lw=1)#, label='Shear Surface')

        ax_surf.fill_between(self.xb/self.norm, self.bathb/self.norm, (self.zetab - self.h0*self.Xi0)/self.norm, alpha=0.4, color='blue', label='Sheared region')
        ax_surf.fill_between(self.xb/self.norm, (self.zetab - self.h0*self.Xi0)/self.norm, self.zetab/self.norm, alpha=0.4, color='red', label='Plug region')

        if self.x_left is not None:
            mask_left_trap = (self.xb >= self.x_left) & (self.xb <= self.x_flat_left)
            ax_surf.fill_between(self.xb[mask_left_trap]/self.norm, 0, self.bathb[mask_left_trap]/self.norm, 
                                alpha=0.5, color='lightpink', hatch='////', edgecolor='salmon')
            # Central rectangle (grey)
            mask_flat_trap = (self.xb >= self.x_flat_left) & (self.xb <= self.x_flat_right)
            ax_surf.fill_between(self.xb[mask_flat_trap]/self.norm, 0, self.bathb[mask_flat_trap]/self.norm, 
                                alpha=0.5, color='grey')
            # Right triangle (pink with hatching pattern)
            mask_right_trap = (self.xb >= self.x_flat_right) & (self.xb <= self.x_right)
            ax_surf.fill_between(self.xb[mask_right_trap]/self.norm, 0, self.bathb[mask_right_trap]/self.norm, 
                                alpha=0.5, color='lightpink', hatch='////', edgecolor='salmon', label = 'Dead region')
        else:    
            ax_surf.fill_between(self.xb/self.norm, 0, self.bathb/self.norm, 
                        alpha=0.5, color='grey')

        ax_surf.set_title('Free Surface & Shear Surface', fontsize=10)
        ax_surf.set_xlabel('x/h0')#, fontsize=12)
        ax_surf.set_ylabel('z/h0')#, fontsize=12)
        #ax_surf.tick_params(axis='both', which='major', labelsize=11)
        #ax_surf.set_xlim([-3.5, 2.5])
        #ax_surf.set_ylim([0, 2])
        ax_surf.legend(loc='upper right', fontsize='small')
        ax_surf.grid(True, linestyle='--')
        
        # Top Right: Normalized Thickness, Velocity, Stress
        ax_flow = axs[0, 1]
        line_h, = ax_flow.plot(self.xb, self.hb/self.norm, 'k-', lw=1, label='h/h0')
        line_u, = ax_flow.plot(self.xb, self.ub/self.u0, 'r-', lw=1, label='u/u0')
        line_tau, = ax_flow.plot(self.xb, self.tb0*np.ones(self.xb.shape)/self.tb0, 'g-', lw=1, label='tau/tb0')
        ax_flow.set_title('Normalized Flow Variables', fontsize=10)
        ax_flow.set_xlabel('x/h0')
        ax_flow.legend(loc='upper right', fontsize='small')
        ax_flow.grid(True, linestyle='--')

        ax_uh = axs_dom[1]
        # Third subplot: Error norms
        ax_err = axs_dom[2]
        line_L1, = ax_err.plot([], [], 'r-', lw=1, label='L1')
        line_L2, = ax_err.plot([], [], 'b-', lw=1, label='L2')
        line_Linf, = ax_err.plot([], [], 'g-', lw=1, label='L∞')
        ax_err.set_title('Error Norms Over Time', fontsize=10)
        ax_err.set_xlabel('Time')
        ax_err.set_ylabel('Error')
        ax_err.legend(loc='upper right', fontsize='small')
        ax_err.grid(True, linestyle='--')
        line_h_dom, = ax_uh.plot(self.xb/self.norm, self.hb/self.norm, 'b-', lw=1, label='h/h0')
        line_u_dom, = ax_uh.plot(self.xb/self.norm, self.ub/self.u0, 'r-', lw=1, label='u/u0')
        line_q_dom, = ax_uh.plot(self.xb/self.norm, (self.hb*self.ub)/self.q0, 'g-', lw=1, label='q/q0')
        ax_uh.set_title('Normalized Flow Variables', fontsize=10)
        ax_uh.set_xlabel('x/h0')
        ax_uh.legend(loc='upper right', fontsize='small')
        ax_uh.grid(True, linestyle='--')
        
        # Bottom Left: Fr and Re
        Fr0, Re0, Bi0, Pl0 = self.get_dimensionless_numbers()

        ax_dim1 = axs[1, 0]
        ax_dim1_r = ax_dim1.twinx()
        line_Fr, = ax_dim1.plot(self.xb, Fr0, 'r-', lw=1, label='Fr')
        line_Re, = ax_dim1_r.plot(self.xb, Re0, 'b-', lw=1, label='Re')
        ax_dim1.set_xlabel('x/h0')
        ax_dim1.set_ylabel('Fr', color='r')
        ax_dim1_r.set_ylabel('Re', color='b')
        ax_dim1.set_title('Froude & Reynolds', fontsize=10)
        ax_dim1.grid(True, linestyle='--')
        
        # Bottom Right: Bi and Pl
        ax_dim2 = axs[1, 1]
        ax_dim2_r = ax_dim2.twinx()
        line_Bi, = ax_dim2.plot(self.xb, Bi0, 'g-', lw=1, label='Bi')
        line_Pl, = ax_dim2_r.plot(self.xb, Pl0, 'm-', lw=1, label='Pl')
        ax_dim2.set_xlabel('x/h0')
        ax_dim2.set_ylabel('Bi', color='g')
        ax_dim2_r.set_ylabel('Pl', color='m')
        ax_dim2.set_title('Bingham & Plastic', fontsize=10)
        ax_dim2.grid(True, linestyle='--')
        
        # Bottom Middle: Lambda
        ax_lam = axs[1, 2]
        Lbd0 = self.Lambda(self.zetab, self.xb)
        line_lam, = ax_lam.plot(self.xb, Lbd0, 'c-', lw=1, label='Lambda')
        ax_lam.axhline(y=self.Lambda0, color='r', linestyle='--', lw=1,  label='Lambda0')
        ax_lam.set_xlabel('x/h0')
        ax_lam.set_ylabel('Lambda', color='c')
        ax_lam.set_title('Lambda', fontsize=10)
        ax_lam.legend(loc='upper right', fontsize='small')
        ax_lam.grid(True, linestyle='--')
        
        # Top Right: Xi
        ax_Xi = axs[0, 2]
        hb0 = np.maximum(self.zetab - self.bathb, 1e-12)
        Lbd0_Xi = self.Lambda(self.zetab, self.xb)
        Xi0,_ = self.Xi(hb0, Lbd0_Xi, self.u)
        line_Xi, = ax_Xi.plot(self.xb, Xi0, 'orange', lw=1, label='Xi')
        ax_Xi.axhline(y=self.Xi0, color='r', linestyle='--', lw=1, label='Xi0')
        ax_Xi.set_xlabel('x/h0')
        ax_Xi.set_ylabel('Xi', color='orange')
        ax_Xi.set_title('Xi', fontsize=10)
        ax_Xi.legend(loc='upper right', fontsize='small')
        ax_Xi.grid(True, linestyle='--')
        
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        # fig.show()
        fig_dom.tight_layout(rect=[0, 0, 1, 0.98])
        # fig_dom.show()

        figs = [fig, fig_dom]
        axes = [ax_surf, ax_flow, ax_dim1, 
            ax_dim1_r, ax_dim2, ax_dim2_r, ax_lam, ax_Xi, ax_dom, ax_uh, ax_err]
        lines = [line_bath, line_zeta, line_plug, line_h,
             line_u, line_tau, line_Fr, line_Re, line_Bi, line_Pl,
             line_lam, line_Xi,
             bath_dom, zeta_dom, plug_dom, line_h_dom, line_u_dom, line_q_dom,
             line_L1, line_L2, line_Linf]

        return figs, axes, lines 


    def _update_plots(self, figs, lines, axes, n, t, errors, status_msg=""):
        global l
        fig, fig_dom = figs
        ax_surf, ax_flow, ax_dim1, ax_dim1_r, ax_dim2, ax_dim2_r, ax_lam, ax_Xi, ax_dom, ax_uh, ax_err = axes
        line_bath, line_zeta, line_plug, line_h, line_u, line_tau, \
            line_Fr, line_Re, line_Bi, line_Pl, line_lam, line_Xi,\
            bath_dom, zeta_dom, plug_dom, line_h_dom, line_u_dom, line_q_dom, line_L1, line_L2, line_Linf = lines

        # Update error norm plot
        self.err_t.append(t)
        self.err_L1.append(errors[0])
        self.err_L2.append(errors[1])
        self.err_Linf.append(errors[2])
        line_L1.set_data(self.err_t, self.err_L1)
        line_L2.set_data(self.err_t, self.err_L2)
        line_Linf.set_data(self.err_t, self.err_Linf)
        ax_err.relim()
        ax_err.set_yscale('log')
        ax_err.autoscale_view()

        zetab = np.where(self.zetab < self.bathb, self.bathb, self.zetab)
        hb = np.maximum(zetab - self.bathb, 0)

        if self.grad == True:
            Lambdab = self.Lambda(self.bathb + hb, self.xb)
        else:
            Lambdab = self.Lambda(self.bathb, self.xb)
            
        Xib, plastic = self.Xi(hb, Lambdab, self.ub) 

        plugheight = np.maximum(np.minimum((zetab - Xib * hb), zetab), self.bathb)

        if self.surge and self.reservoir:
            plugheight = np.where(plastic, self.bathb, plugheight)
        # ----------------------------------------------------------
        # Calculate stress
             
        ub_cent = self.ub
        taub = self.tau(Xib, ub_cent, hb)
        #taub = np.where(taub < 0, self.ty, taub)

        if self.reservoir:
            x_norm = self.xb # / l
            denom = 1 
            tnorm = 1 
        else:
            x_norm = self.xb / self.norm
            denom = self.norm
            tnorm = 1



        # ----------------------------------------------------------
        # Free surface plot
        line_bath.set_data(x_norm, self.bathb/denom)
        line_zeta.set_data(x_norm, zetab/denom)
        line_plug.set_data(x_norm, plugheight/denom)

        for collection in ax_surf.collections:
            if collection.get_alpha() == 0.4: 
                collection.remove()
            
        ax_surf.fill_between(x_norm, self.bathb/denom, plugheight/denom, alpha=0.4, color='blue', label='Sheared region')
        ax_surf.fill_between(x_norm, plugheight/denom, zetab/denom, alpha=0.4, color='red', label='Plug region')
        ax_surf.relim()
        ax_surf.autoscale_view()
        ax_surf.set_ylim(0, np.max(zetab/denom)*1.5) 
        ax_surf.set_xlim(-2 + self.center/denom, 2 + self.center/denom)
        ax_surf.legend(loc = 'best', fontsize='small')

        # ----------------------------------------------------------
        # Domain overview plot
        bath_dom.set_data(x_norm, self.bathb/denom)
        zeta_dom.set_data(x_norm, zetab/denom)
        plug_dom.set_data(x_norm, plugheight/denom)
        ax_dom.relim()
        ax_dom.set_xlim(self.xL/denom*1.1, self.xR/denom*1.1)
        ax_dom.autoscale_view()
        # ax_dom.set_xlim([-25, 25])
        # ax_dom.set_ylim([0, 6])



        line_h_dom.set_data(x_norm, self.hb/self.h0)
        line_u_dom.set_data(x_norm, ub_cent/self.u0)
        line_q_dom.set_data(x_norm, (self.hb*ub_cent)/self.q0)
        ax_uh.relim()
        ax_uh.autoscale_view()
        ax_uh.set_xlim(self.xL/denom*1.1, self.xR/denom*1.1)

        # ----------------------------------------------------------
        # Depth, velocity and basal stress plot
        line_h.set_data(x_norm, self.hb/self.h0)
        line_u.set_data(x_norm, ub_cent/self.u0)
        line_tau.set_data(x_norm, taub/self.tb0)
        ax_flow.relim()
        ax_flow.autoscale_view()
        ax_flow.set_xlim(-5 + self.center, 5 + self.center)
        if np.max(taub/self.tb0 > 10):
            ax_flow.set_yscale('log')
        else:
            ax_flow.set_yscale('linear')
        
        # ----------------------------------------------------------
        # Dimensionless numbers plots 
        Fr, Re, Bi, Pl = self.get_dimensionless_numbers()
        
        line_Fr.set_data(x_norm, Fr)
        line_Re.set_data(x_norm, Re)
        ax_dim1.relim()
        ax_dim1.autoscale_view()
        ax_dim1_r.relim()
        ax_dim1_r.autoscale_view()
        ax_dim1_r.set_xlim(-5 + self.center, 5 + self.center)
        
        line_Bi.set_data(x_norm, Bi)
        line_Pl.set_data(x_norm, Pl)
        ax_dim2.relim()
        ax_dim2.autoscale_view()
        ax_dim2_r.relim()
        ax_dim2_r.autoscale_view()
        ax_dim2_r.set_xlim(-5 + self.center, 5 + self.center)
        
        # Update Lambda plot
        if self.grad:
            Lbd = self.Lambda(self.zetab, self.xb)
        else:
            Lbd = self.Lambda(self.bathb, self.xb)
        line_lam.set_data(x_norm, Lbd)
        ax_lam.relim()
        ax_lam.autoscale_view()
        ax_lam.set_xlim(-5 + self.center, 5 + self.center)
        
        # Update Xi plot
        Xib_plot,_ = self.Xi(self.hb, Lbd, self.ub)
        line_Xi.set_data(x_norm, Xib_plot)
        ax_Xi.relim()
        ax_Xi.autoscale_view()
        ax_Xi.set_xlim(-5 + self.center, 5 + self.center)


        if self.reservoir:
            for ax in axes:
                ax.set_xlim(self.xL, 2.8)
                if ax == ax_surf:
                    ax.set_ylim(0, 0.1)
        else:
            pass

        # ----------------------------------------------------------
        # Update title BEFORE drawing
        fig.suptitle(f't = {t/tnorm:.3f}, step = {n}, CFL = {self.CFL:.2f}\n' \
        f'ty = {self.ty:.3f} Pa, m = {self.m:.3f}, kn = {self.kn:.3f} Pa.s^n, rho = {self.rho} kg/m³\n h0 = {self.h0:.3f}, theta = {np.degrees(self.theta):.1f}° M = {self.M:.3f}, alpha = {self.M/self.h0:.3f} || Nx = {self.Nx}, dx = {self.dx:.3e}, dt = {self.dt:.3e}\n' \
        f'Errors: L1 = {errors[0]:.3e}, L2 = {errors[1]:.3e}, Linf = {errors[2]:.3e}', fontsize=10, y=0.98) 

        fig_dom.suptitle(f'Domain Overview - t = {t:.3f}, step = {n}\n', fontsize=10)

        fig.canvas.draw()
        fig_dom.canvas.draw()
        plt.pause(1e-12)

        return


    def solve(self, plot_interval=10, max_iter=1e5, probes=None, live_plot=True):
        # Start timing the simulation
        start_time = time.time()
        
        self.tic = 0.0
        t = 0
        n = 0
        
        # Setup probes
        probe_indices = []
        if probes is not None:
            for p in probes:
                idx = (np.abs(self.xb - p)).argmin()
                probe_indices.append(idx)
            
            self.time_series = {
                't': [],
                'h': {p: [] for p in probes},
                'u': {p: [] for p in probes}
            }

        # Initialize time series storage for surface profiles and front positions
        if self.surge:
            self.surface_profiles = {'times': [], 'profiles': [], 'plugheights': [], 'x': self.xb.copy()}
            self.front_positions = {'times': [], 'positions': []}
            profile_save_interval = max(1, plot_interval // 5)  # Save profiles 5x more frequently than plots

        # Setup live plotting
        if live_plot:
            self.err_t, self.err_L1, self.err_L2, self.err_Linf = [], [], [], []
            #plt.ion()
            figs, axes, lines = self._initialize_plots()

        # Main loop
        for n in range(int(max_iter)):
            zetab_old = self.zetab.copy()

            lamb1 = self.ub + np.sqrt(self.g * self.hb * np.cos(self.theta))
            lamb2 = self.ub - np.sqrt(self.g * self.hb * np.cos(self.theta))
            
            self.dt = self.CFL * self.dx / np.max(np.abs([lamb1, lamb2]))

            if t + self.dt > self.tend:
                self.dt = self.tend - t
            if t >= self.tend:
                break
            
            # Save surface profile at regular intervals for animation
            if self.surge:
                if n % profile_save_interval == 0:
                    self.surface_profiles['times'].append(t)
                    self.surface_profiles['profiles'].append(self.zetab.copy())
                    
                    # Calculate and save plug height (shear surface)
                    if self.grad:
                        Lbd_temp = self.Lambda(self.zetab, self.xb)
                    else:
                        Lbd_temp = self.Lambda(self.bathb, self.xb)

                    Xib, plastic = self.Xi(self.hb, Lbd_temp, self.ub) 

                    plugheight = np.maximum(np.minimum((self.zetab - Xib * self.hb), self.zetab), self.bathb)

                    plugheight = np.where(plastic, self.bathb, plugheight)

                    self.surface_profiles['plugheights'].append(plugheight.copy())
                    
                    # Track front position (find rightmost point with significant flow depth)
                    wet_mask = self.hb > 1e-10
                    if np.any(wet_mask):
                        front_idx = np.where(wet_mask)[0][-1]
                        self.front_positions['times'].append(t)
                        self.front_positions['positions'].append(self.xb[front_idx])

            # Compute the coefficients of the linear part and the r.h.s
            self.hb = np.maximum(self.zetab - self.bathb, 1e-12)
            Hxm, rhs = self._LinearPartCoeff(self.zetab, self.hb, self.u)

            # Explicit step ----------------------------------------------------
            ustar = self._MomentumConvection(self.u) 
            # ------------------------------------------------------------------
            
            # Implicit step ----------------------------------------------------
            # Solve (I + M) * zetab = rhs + bathb
            maxiter = 1000
            tol = 1e-8
            if self.surge:
                for k in range(maxiter):
                    Hb = np.maximum(self.zetab - self.bathb, 1e-12)  
                    HM_eta = Hb + self._MatVectProd_zeta(self.zetab, Hxm)  
                    wet = Hb > 1e-10
                    residual = HM_eta - rhs   # Residual of equation (33)
                    residual_norm = np.linalg.norm(residual)
                    if residual_norm < tol:
                        break
                    delta_zetab = self._CGsolver(residual, Hxm, wet)
                    self.zetab = self.zetab - delta_zetab
            else:
                wet = np.ones(self.Nx, dtype=bool)
                self.zetab = self._CGsolver(rhs + self.bathb, Hxm, wet)
            
            # ------------------------------------------------------------------

            self.hb = np.maximum(self.zetab - self.bathb, 1e-12)

            # ------------------------------------------------------------------
            self.u = self._VelocityUpdate(ustar , self.zetab, self.hb)

            # ------------------------------------------------------------------
            
            t += self.dt

            # Record probes
            if probes is not None:
                self.time_series['t'].append(t)
                for i, p in enumerate(probes):
                    idx = probe_indices[i]
                    self.time_series['h'][p].append(self.hb[idx])
                    self.time_series['u'][p].append(self.ub[idx])

            # Compute convergence criterion
            self.ub = 0.5*(self.u[0:self.Nx] + self.u[1:self.Nx+1])
            self.hb = np.maximum(self.zetab - self.bathb, 1e-12)
            
            if n % plot_interval == 0:
                print(f"Step {n}, t={t:.3f}")

                if live_plot:
                    diff = self.zetab - zetab_old
                    errors = (
                        np.mean(np.abs(diff)),
                        np.sqrt(np.mean(diff**2)),
                        np.max(np.abs(diff)),
                    )
                    self._update_plots(figs, lines, axes, n, t, errors)
                
            if self.roll:
                pass
            elif self.reservoir:
                if self.grad:
                    Xi, Xi_check = self.Xi(self.hb, self.Lambda(self.zetab, self.xb), self.ub)
                else:
                    Xi, Xi_check = self.Xi(self.hb, self.Lambda(self.bathb, self.xb), self.ub)
                
                plastic = self.Plastic(self.ub, self.hb)
                l2 = lambda f: np.sqrt(np.sum((1-f)**2))

                if l2(Xi[wet]) <= 0.01 and l2(plastic[wet]) <= 0.01:
                    x_last = self.xb[np.where(self.hb > 1e-6)[0][-1]]
                    print(f"Flow stopped at step {n}, t={t:.3f}, x stop={x_last}")
                    break

        # Post-processing results
        if self.grad:
            Lbd = self.Lambda(self.zetab, self.xb)
        else:
            Lbd = self.Lambda(self.bathb, self.xb)

        Xib,_ = self.Xi(self.hb, Lbd, self.ub)
        taub = self.tau(Xib, self.ub, self.hb)
        plugheight = np.maximum(np.minimum((self.zetab - Xib * self.hb), self.zetab), self.bathb)        

        Fr, Re, Bi, Pl = self.get_dimensionless_numbers()
        
        results = {
            'xb': self.xb,
            'zetab': self.zetab,
            'bathb': self.bathb,
            'plugheight': plugheight,
            'tau': taub,
            'hb': self.hb,
            'ub': self.ub,
            'u': self.u,
            'Fr': Fr,
            'Re': Re,
            'Bi': Bi,
            'Pl': Pl,
            'tb0': self.tb0,
            # Add parameters for post-processing
            'Xi0': self.Xi0,
            'ty': self.ty,
            'm': self.m,
            'kn': self.kn,  
            'rho': self.rho,
            'g': self.g,
            'theta': self.theta,
            'M': self.M,
            'h0': self.h0,
            'u0': self.u0
        }

        if probes is not None:
            results['time_series'] = self.time_series
        
        # Always include surface profiles and front positions
        if self.surge:
            results['surface_profiles'] = self.surface_profiles
            results['front_positions'] = self.front_positions
        
        # Calculate and store simulation duration
        end_time = time.time()
        simulation_duration = end_time - start_time
        results['simulation_duration'] = simulation_duration
        print(f"Total simulation time: {simulation_duration:.2f} seconds")


        # if live_plot:
        #     plt.ioff()
            
        return results
    

    def run(self, bathymetry=None, x_left=None, x_right=None, x_flat_left=None, x_flat_right=None, plot_interval=10, max_iter=1e5, probes=None, center=0, norm = None, grad = True, zeta_file=None, u_file=None, live_plot=True):
        """
        Runs the solver using the bathymetry/region markers already set via
        set_bathymetry (or explicit overrides passed here).
        """
        self.center = center
        self.grad = grad

        if norm == None:
            self.norm = self.h0
        else:
            self.norm = norm

        bathymetry = self.bathb if bathymetry is None else bathymetry
        x_left = self.x_left if x_left is None else x_left
        x_right = self.x_right if x_right is None else x_right
        x_flat_left = self.x_flat_left if x_flat_left is None else x_flat_left
        x_flat_right = self.x_flat_right if x_flat_right is None else x_flat_right

        self.set_case(bathymetry, x_left, x_right, x_flat_left, x_flat_right, zeta_file=zeta_file, u_file=u_file)
        
        results = self.solve(plot_interval=plot_interval, max_iter=max_iter, probes=probes, live_plot=live_plot)

        return results
