"""
See evolved NEAT ANN
"""

from __future__ import print_function

import os
import pickle

import model

import neat
import visualize
import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

# load the winner
foldername = ''
network = ''
reduced_inputs = 0
# with open('C:\\EK_Projects\\DS_3DOF\\winner', 'rb') as f:
with open('C:\\EK_Projects\\CP_NEAT\\winner', 'rb') as f:
# with open('C:\\EK_Projects\\CartPole_NEAT\\'+network, 'rb') as f:
    nn = pickle.load(f)
#end
print(nn)

directory = "C:\\EK_Projects\\CP_NEAT\\"+foldername+"\\"

# visualize.plot_stats(stats, ylog=True, view=True, filename="fitness.svg")
# visualize.plot_species(stats, view=True, filename="speciation.svg")

node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'F'}

# visualize.draw_net(config, nn, True, node_names=node_names)'C:\\EK_Projects\\DS_3DOF\\Results\\'+foldername+'\\Individual\\'+name+'.png'
# visualize.draw_net(config, nn, view=True, node_names=node_names,filename=directory+network+"_network.gv")
# visualize.draw_net(config, nn, view=True, node_names=node_names,filename="winner-enabled.gv", show_disabled=False)
visualize.draw_net(config, nn, view=True, node_names=node_names,filename=directory+network+"_network-enabled-pruned.gv", show_disabled=False, prune_unused=True)
