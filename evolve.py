"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function
import os
import pickle
import model
import neat
import visualize
import numpy as np
import plots as myplts

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

runs_per_net = 1
generations = 2

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        
        sim = model.CartPole()                                 # create new cart pole instance
        done = False                                           # flag to indicate end of episode
        
        # Run the given simulation for up to num_steps time steps.
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
            
        # Evaluate genome fitness
        fitness = sim.get_fitness()
        fitnesses.append(fitness)
    
    # The genome's fitness is its worst performance across all runs (if multiple runs) or simply its single-run performance (if only 1 run)
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def main():
    # Load the config file, which is assumed to live in the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate,generations)

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

    # Show winning neural network
    print(winner)

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'F'}
    visualize.draw_net(config, winner, view=True, node_names=node_names,filename="winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    ########################
    # Simulate the performance of the winner
    ########################
    net = neat.nn.FeedForwardNetwork.create(winner, config) # load winner network
    sim = model.CartPole()
        
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

    print(f'Pole balanced for {round(sim.t,1)} of {sim.simtime} seconds.')
    myplts.states(sim)

if __name__ == '__main__':
    main()