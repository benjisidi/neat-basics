import os
from copy import copy
import neat
import numpy as np


def random_sign():
    return -1 if np.random.random() > 0.5 else 1


def random_value():
    return np.random.random() * random_sign()


def generate_bandit_inputs():
    """
    We want all 3 cases to be equally likely.
    """
    winning_case = np.random.randint(0, 3)
    # Case 0: x0>0, others can be any
    if winning_case == 0:
        x0 = np.random.random()
        x1 = random_value()
        x2 = random_value()
    elif winning_case == 1:
        x0 = -np.random.random()
        x1 = random_value()
        x2 = -x1 + np.random.random()
    else:
        x0 = -np.random.random()
        x1 = random_value()
        x2 = -x1 - np.random.random()
    return (x0, x1, x2)


def eval_bandit_step(xi, choice):
    """
    Simple multi-armed bandit setup:
    Arm 0 best when x0 > 0
    Arm 1 best when x1 + x2 > 0
    Arm 2 best otherwise
    """
    if xi[0] > 0:
        return 1 if choice == 0 else -1
    if xi[1] + xi[2] > 0:
        return 1 if choice == 1 else -1
    else:
        return 1 if choice == 2 else -1


def eval_bandit_play(net, its=100):
    score = 0
    for i in range(its):
        xi = generate_bandit_inputs()
        choice = np.argmax(net.activate(xi))
        it_score = eval_bandit_step(xi, choice)
        score += it_score
    return score


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_bandit_play(net)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="MAB-"))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print(f'\nBest genome:\n{winner!s}')


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'multi_armed_bandit_config')
    run(config_path)
