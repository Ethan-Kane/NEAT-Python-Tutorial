# Cart pole dynamics and modelling

from math import cos, pi, sin
import random
import numpy as np

class CartPole(object):
    
    # Simulation parameters
    simtime = 200                                       # maximum simulation time (s)
    dt = 0.02                                           # time step (s)

    # Physics
    gravity = 9.8                                       # acceleration due to gravity, positive is downward, (m/sec^2)

    # Cart pole
    mcart = 0.711                                       # cart mass (kg)
    mpole = 0.209                                       # pole mass (kg)
    lpole = 0.326                                       # half the pole length (m)
    x_max = 2.4                                         # limits for cart position (m)
    theta_max = np.deg2rad(12)                          # limit for pole angle (rads)

    def __init__(self, x=None, theta=None, dx=None, dtheta=None, force=None):

        # Randomize initial states
        if x is None:
            x = random.uniform(-0.5 * self.x_max, 0.5 * self.x_max)
        if theta is None:
            theta = random.uniform(-0.5 * self.theta_max, 0.5 * self.theta_max)
        if dx is None:
            dx = random.uniform(-1.0, 1.0)
        if dtheta is None:
            dtheta = random.uniform(-1.0, 1.0)
        if force is None:
            force = 15
        
        self.force = force

        self.t = 0.0
        self.x = x
        self.theta = theta
        
        self.dx = dx
        self.dtheta = dtheta
        
        self.ddx = 0.0
        self.ddtheta = 0.0

        # Initialize data lists for analysis
        self.t_list = [self.t]
        self.x_list = [self.x]
        self.theta_list = [self.theta]
        self.dx_list = [self.dx]
        self.dtheta_list = [self.dtheta]
        self.action_list = [np.nan]
    
    def step(self, force):

        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole
        dt = self.dt
                
        # Update position/angle.
        self.x += dt * self.dx
        self.theta += dt * self.dtheta
        
        # Compute new accelerations
        st = sin(self.theta)
        ct = cos(self.theta)
        ddtheta1 = (mt*g*st-ct*(force+st*mp*L*self.dtheta**2))/((4/3)*mt*L-mp*L*ct**2)
        ddx1 = (force+mp*L*(st*self.dtheta**2 - ddtheta1*ct))/mt
        
        # Update velocities.
        self.dx += (ddx1) * dt
        self.dtheta +=  (ddtheta1) * dt
        
        # Remember current acceleration for next step.
        self.ddtheta = ddtheta1
        self.ddx = ddx1
        self.t += dt

        # Record state history
        self.t_list.append(self.t)
        self.x_list.append(self.x)
        self.theta_list.append(self.theta)
        self.dx_list.append(self.dx)
        self.dtheta_list.append(self.dtheta)
        self.action_list.append(force)

    def get_states(self): # continuous (normalized) state space
        states = [0.5*(self.x+self.x_max)/self.x_max,
                  0.5*(self.theta+self.theta_max)/self.theta_max,
                  (self.dx+0.75)/1.5,
                  (self.dtheta+1)/2]
        return states

    def actuator(self,action): # go left when 0, right when 1
        return self.force if np.squeeze(action) >= 0.5 else -self.force

    def get_fitness(self): # calculate fitness
        fitness = self.t_list[-1] # fitness is simply the amount of time for which the pole didn't fall over
        return fitness

    def check_done(self): # check if simulation ends due to states exceeding bounds (ie. pole falling over)
        done = False
        if abs(self.x) >= self.x_max or abs(self.theta) >= self.theta_max:
            done = True
        return done