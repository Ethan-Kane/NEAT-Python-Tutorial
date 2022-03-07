"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function
import os
import pickle
import model
import neat
import plots as myplts
import numpy as np
import animate

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load the winner
with open('./winner','rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
sim = model.CartPole()

########################
# Simulate the performance of the loaded network
########################    
done = False
while not done and sim.t < sim.simtime:
    # Check if done
    done = sim.check_done()
    # Get cartpole states
    states = sim.get_states()
    # Apply inputs to ANN
    action = net.activate(states)
    # Obtain control values
    control = sim.actuator(action)
    sim.step(control)

########################
# Performance summary
########################
print(f'Pole balanced for {round(sim.t,1)} of {sim.simtime} seconds.')
myplts.states(sim)

animate.animate(sim)
