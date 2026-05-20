import numpy as np
import time

class SWEHBSolver:
    def __init__(self, params):
        """
        Initialize the SWE HB Solver with the given parameters.
        params: dictionary containing:
            g, h0, ty, kn, m, rho, theta (fluid properties)
            Nx, CFL, tend (numerical parameters)
            M, xL, xR (domain and obstacle parameters)
        """
        self.dam_break = False
        self.reservoir = False
        self.roll = False
        self.flux = 'Godunov'  # Only Godunov is supported in this lite version
        self.grad = True

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
        self.psi0 = self.ty / (self.rho * self.g * self.h0 * self.Lambda0)
        self.hcrit = self.ty / (self.rho * self.g * self.Lambda0)
        
        term1 = (self.rho * self.g * self.Lambda0 * ((self.h0 * (1 - self.psi0))**(self.m + 1)) / self.kn)**(1/self.m)
        term2 = (1 - self.m / (2 * self.m + 1) * (1 - self.psi0))
        self.u0 = (self.m / (self.m + 1)) * term1 * term2
        self.umax = self.u0*(2*self.m + 1)/(1 + self.m + self.m*self.psi0)
        
        self.Fr0 = self.u0/(np.sqrt(self.g * np.cos(self.theta) * (self.h0**0.5)))

        self.mu0 = self.kn*self.mu(self.psi0)*(self.u0/self.h0)**self.m
        self.tb0 = self.ty + self.mu0
        self.z0 = self.h0 * (1 - self.psi0)
        self.ss = self.u0/self.h0
        self.q0 = self.h0 * self.u0
        
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


    def set_case(self, bathymetry, x_left=None, x_right=None, x_flat_left=None, x_flat_right=None, dam_break=False, reservoir=False, zeta_file = None, u_file = None):
        """
        Sets the case bathymetry and initial conditions according to the case's geometry and type. 
        """
        self.bathb = bathymetry.copy()
        self.dam_break = dam_break

        global l

        def dam_break_reservoir(self, x, h_g):
            global l
            h = np.zeros_like(x)
            mask = (x >= 0) & (x <= l)
            h[mask ] = h_g + (x[mask] - l) * np.tan(self.theta)
            return h

        if self.dam_break:
            self.u = np.zeros(self.Nx + 1)
            self.bathb = bathymetry.copy()
            
            if reservoir:
                l = 0.50
                h_g = self.h0
                self.bathb[self.xb <= 0] = h_g
                self.zetab = dam_break_reservoir(self, self.xb, h_g) + self.bathb

                self.u0 = 0
                
            else:
                # Original step dam break
                self.zetab = np.zeros(self.Nx, dtype=np.float128)
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
    

    def Psi(self, h, Lbd, u = None):
        """Calculate local Psi parameter."""
        h_safe = np.maximum(h, 1e-12)
        psi = self.ty / (self.rho * self.g * h_safe * np.abs(Lbd))
        psi_filter = np.abs(psi) >= 1

        if isinstance(h, np.ndarray) == False:
            return psi, psi_filter
        else:
            pass
        
        if (self.reservoir == False) and (h_safe[-1] > 1e-12):      
            psi = np.where(psi >= 1, self.psi0, psi)
            if u is not None:
                if u.size != h.size:
                    u = 0.5*(u[0:self.Nx] + u[1:self.Nx+1])
                Bi = self.Bingham(u, h)
                Fr = self.Froude(u, h)
                psi = np.where((Bi < 1) & (Fr > 1), 0, psi)  
                return psi, psi_filter 
            else:
                return psi, psi_filter
        
        else:
            psi = np.where(psi >= 1, 1, psi)
            return psi, psi_filter


    def Lambda(self, f, x):
        grad_f = np.gradient(f, x)
        l = np.sin(self.theta) - grad_f * np.cos(self.theta) 
    
        return l


    def mu(self, psi):
        """Calculate effective viscosity multiplier."""

        psi = np.minimum(psi, 0.99)  # or 1 - 1e-6
        psi_alt = ((1 + 2*self.m)/(1 + self.m*(1 + psi))*(1 + self.m)/(self.m*(1 - psi)))
        psi_alt = np.maximum(psi_alt, 0)  
        muv = psi_alt**self.m

        return muv


    def tau(self, psi, u, h):
        """Calculate basal shear stress."""
        h_safe = np.maximum(h, 1e-12)
        ss = u / h_safe
        t = (self.ty + self.kn * self.mu(psi) * ((np.abs(ss))**self.m))
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
        psim = np.zeros(self.Nx, dtype=np.float128)
        psip = np.zeros(self.Nx, dtype=np.float128)
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

        psim,_ = self.Psi(Hxm, Lambda_m, u[0:self.Nx])
        psip,_ = self.Psi(Hxp, Lambda_p, u[1:self.Nx+1])

        mup = self.kn * self.mu(psip)
        mum = self.kn * self.mu(psim)
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


    def ExactRiemannProblem(self, qL, qR, xi = 0):
        f = lambda q: 0.5 * q**2
        
        if qL > qR:
            shock_speed = (f(qR) - f(qL)) / (qR - qL)
            
            if xi <= shock_speed:
                q = qL
            else:
                q = qR
                
        else:          
            if qL > 0:
                q = qL
            elif qR < 0:
                q = qR
            else:
                q = xi
        
        return q
    

    def _MomentumConvection(self, u):
        ustar = u.copy()
        dtdx = self.dt / self.dx
        self.tic += self.dt
        amp = 0.02
        freq = 1.5

        f = lambda q : 0.5 * q**2

        # Set boundary conditions
        if self.reservoir:
            ustar[0] = ustar[0]
            
        elif self.roll:
            ustar[0] = self.u0*(1 + amp*np.sin(2 * np.pi * self.tic * freq))
        else:
            ustar[0] = self.u0

        # Godunov flux scheme
        for i in range(1, self.Nx):
            qm = self.ExactRiemannProblem(u[i-1], u[i], 0)
            Fm = f(qm)
            qp = self.ExactRiemannProblem(u[i], u[i+1], 0)
            Fp = f(qp)
            
            ustar[i] -= dtdx * (Fp - Fm)
        
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

        psip, plastic = self.Psi(h_itf, Lambdab, u)
        Pl = self.Plastic(u, h_itf)

        mu = self.kn*self.mu(psip)
        ss_alt = u/h_itf

        #taub = self.tau(psip, u, hp)/self.rho
        taub_itf = 1/self.rho*(self.ty + mu*((np.abs(ss_alt))**self.m))

        u[1:self.Nx] = u[1:self.Nx] \
            - self.g*self.dt/self.dx*np.cos(self.theta)*(zetab[1:self.Nx] - zetab[0:self.Nx-1]) \
            + self.dt*self.g*np.sin(self.theta)\
            - self.dt/h_itf[1:self.Nx]*taub_itf[1:self.Nx]
        
        u[self.Nx] = u[self.Nx - 1]  
        
        u = np.where(u < 1e-9, 0, u)
        

        return u
    

    def solve(self, plot_interval=10, max_iter=1e5, probes=None):
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
        if self.dam_break:
            self.surface_profiles = {'times': [], 'profiles': [], 'plugheights': [], 'x': self.xb.copy()}
            self.front_positions = {'times': [], 'positions': []}
            profile_save_interval = max(1, plot_interval // 5)  # Save profiles 5x more frequently than plots

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
            if self.dam_break:
                if n % profile_save_interval == 0:
                    self.surface_profiles['times'].append(t)
                    self.surface_profiles['profiles'].append(self.zetab.copy())
                    
                    # Calculate and save plug height (shear surface)
                    if self.grad:
                        Lbd_temp = self.Lambda(self.zetab, self.xb)
                    else:
                        Lbd_temp = self.Lambda(self.bathb, self.xb)

                    psib, plastic = self.Psi(self.hb, Lbd_temp, self.ub) 

                    plugheight = np.maximum(np.minimum((self.zetab - psib * self.hb), self.zetab), self.bathb)

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

            # Explicit step
            ustar = self._MomentumConvection(self.u) 
            
            # Implicit step
            # Solve (I + M) * zetab = rhs + bathb
            maxiter = 1000
            tol = 1e-8
            if self.dam_break:
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
            
            self.hb = np.maximum(self.zetab - self.bathb, 1e-12)

            self.u = self._VelocityUpdate(ustar , self.zetab, self.hb)
            
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
                
            if self.roll:
                pass
            elif self.reservoir:
                if self.grad:
                    psi, psi_check = self.Psi(self.hb, self.Lambda(self.zetab, self.xb), self.ub)
                else:
                    psi, psi_check = self.Psi(self.hb, self.Lambda(self.bathb, self.xb), self.ub)
                
                plastic = self.Plastic(self.ub, self.hb)
                l2 = lambda f: np.sqrt(np.sum((1-f)**2))

                if l2(psi[wet]) <= 0.01 and l2(plastic[wet]) <= 0.01:
                    x_last = self.xb[np.where(self.hb > 1e-6)[0][-1]]
                    print(f"Flow stopped at step {n}, t={t:.3f}, x stop={x_last}")
                    break

        # Post-processing results
        if self.grad:
            Lbd = self.Lambda(self.zetab, self.xb)
        else:
            Lbd = self.Lambda(self.bathb, self.xb)

        psib,_ = self.Psi(self.hb, Lbd, self.ub)
        taub = self.tau(psib, self.ub, self.hb)
        plugheight = np.maximum(np.minimum((self.zetab - psib * self.hb), self.zetab), self.bathb)        

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
            'psi0': self.psi0,
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
        if self.dam_break:
            results['surface_profiles'] = self.surface_profiles
            results['front_positions'] = self.front_positions
        
        # Calculate and store simulation duration
        end_time = time.time()
        simulation_duration = end_time - start_time
        results['simulation_duration'] = simulation_duration
        print(f"Total simulation time: {simulation_duration:.2f} seconds")
            
        return results
    

    def run(self, bathymetry, x_left=None, x_right=None, x_flat_left=None, x_flat_right=None, plot_interval=10, max_iter=1e5, probes=None, dam_break=False, reservoir=False, roll=False, flux='Godunov', center=0, norm = None, grad = True, zeta_file=None, u_file=None):

        self.roll = roll
        self.flux = 'Godunov' # Enforce Godunov
        self.dam_break = dam_break
        self.reservoir  = reservoir
        self.center = center
        self.grad = grad

        if norm == None:
            self.norm = self.M
        else:
            self.norm = norm

        self.set_case(bathymetry, x_left, x_right, x_flat_left, x_flat_right, dam_break=dam_break, reservoir=reservoir, zeta_file=zeta_file, u_file=u_file)
        
        results = self.solve(plot_interval=plot_interval, max_iter=max_iter, probes=probes)

        return results