import matplotlib.pyplot as plt
import numpy as np
from numpy import square as sq

plt.rcParams['axes.grid'] = True

def states(sim):
    fig,axs = plt.subplots(3,2)
    plt.subplots_adjust(hspace = 0.5)
    plt.title('Cart pole states')

    axs[0,0].plot(sim.t_list,sim.x_list)
    axs[0,0].set_ylabel('x, m')

    axs[0,1].plot(sim.t_list,sim.dx_list)
    axs[0,1].set_ylabel(r'$\dot{x}$, m/s')

    axs[1,0].plot(sim.t_list,np.rad2deg(sim.theta_list))
    axs[1,0].set_ylabel(r'$\theta$, deg')

    axs[1,1].plot(sim.t_list,np.rad2deg(sim.dtheta_list))
    axs[1,1].set_ylabel(r'$\dot{\theta}$, deg/s')

    axs[2,0].plot(sim.t_list,sim.action_list)
    axs[2,0].set_ylabel('Force, N')

    for i in range(3):
        for j in range(2):
            axs[i,j].set_xlabel('Time, s')

    plt.show()