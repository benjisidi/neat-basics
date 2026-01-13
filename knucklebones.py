from collections import Counter
import os
from copy import copy
import neat
import numpy as np
from itertools import combinations
from tqdm import tqdm
import math

# EL PLAN
"""
Each cell is represented by a 7-length vector meaning [EMPTY, 1, ..., 6]
Gonna hold onto these separately and concat them when they get passed to nets
So game state will be BOARD_A, BOARD_B
And net A gets passed (*BOARD_A, *BOARD_B), while B gets passed (*BOARD_B, *BOARD_A)
Board is col1, col2, col3 (front middle back doesn't actually matter)

Functions:

# These function on a whole board at once: plan is to keep game-state as ndarray and
# convert to one-hot when passing to nets
itoh: Integer to One-Hot
process_net_output: Takes the (one-hot) output of the net and returns a sorted list of (integer) indices for where it would like to place the current die
board2input: One-hots both boards and concats them
n_populated(board)->number: counts how many dice are on a board so we know when to end the game
score(board)->number: totals up the score of a board
process_move(board1, board2, value, loc)->(board1, board2): process placement of value in loc on board1, update both boards as required

"""


def get_net_input(board1: np.ndarray, board2: np.ndarray, roll: int) -> np.ndarray:
    output = np.zeros(132)
    for i, val in enumerate(board1):
        idx = i*7 + val
        output[idx] = 1
    for i, val in enumerate(board2):
        idx = i*7 + val + 63
        output[idx] = 1
    roll_idx = 126 + roll - 1
    output[roll_idx] = 1
    return output


def process_net_output(boards: list[np.ndarray], move_preferences: list[float],
                       turn_counter: int, opponent_index: int, roll: int) -> list[np.ndarray]:
    # First, find the most preferred legal move by iterating through the preferences til a legal one is found
    player_board = boards[turn_counter]
    desired_move = -1
    # Argsort is always descending, so we'll reverse the arry to get the most preferred moves first
    move_preference_indices = np.argsort(move_preferences)[::-1]
    for move in move_preference_indices:
        if player_board[move] == 0:
            desired_move = move
            break
    if desired_move == -1:
        raise ValueError(
            "No valid moves. Are you trying to play when the game is over?")
    # Next, place the roll in the empty space of the current player's board
    player_board[desired_move] = roll
    # Last, blank any matching rolls from the opposing player's column
    opponent_board = boards[opponent_index]
    column = desired_move // 3
    column_indices = np.array([0, 1, 2]) + 3 * column
    for idx in column_indices:
        if opponent_board[idx] == roll:
            opponent_board[idx] = 0
    return [player_board, opponent_board] if turn_counter == 0 else [opponent_board, player_board]


def get_blank_board() -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])


def is_game_over(boards: list[np.ndarray]) -> bool:
    # Game is over if either board is full, this equates to all values being
    # non-zero since 0 is blank here
    if np.all(boards[0]) or np.all(boards[1]):
        return True
    return False


def sum_col(col: np.ndarray) -> int:
    total = 0
    counts = Counter(col)
    for i in col:
        total += i * counts[i]
    return total


def get_scores(boards: list[np.ndarray]) -> list[int]:
    scores = [0, 0]
    for i, board in enumerate(boards):
        cols = np.array_split(board, 3)
        total = np.sum([sum_col(col) for col in cols])
        scores[i] = total
    return scores


def play_game(net_a: neat.nn.FeedForwardNetwork, net_b: neat.nn.FeedForwardNetwork):
    nets = [net_a, net_b]
    boards = [get_blank_board(), get_blank_board()]
    turn_counter = np.random.randint(0, 2)
    game_over = False
    turns = 0
    max_turns = 200
    while not game_over and turns < max_turns:
        current_net = nets[turn_counter]
        opponent_index = (turn_counter + 1) % 2
        roll = np.random.randint(1, 7)
        move_preferences = current_net.activate(get_net_input(
            boards[turn_counter], boards[opponent_index], roll))
        boards = process_net_output(
            boards, move_preferences, turn_counter, opponent_index, roll)
        game_over = is_game_over(boards)

        turn_counter = opponent_index
        turns += 1
    scores = get_scores(boards)
    return scores


def eval_genomes(genomes, config):
    # We're goint to run a round-robin tournament between all of our genomes
    for pairing in tqdm(combinations(genomes, 2), total=math.comb(len(genomes), 2)):
        # for genome_id, genome in genomes:
        [(id_a, genome_a), (id_b, genome_b)] = pairing
        net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)
        net_b = neat.nn.FeedForwardNetwork.create(genome_b, config)
        score_a, score_b = play_game(net_a, net_b)
        if score_a > score_b:
            genome_a.fitness = 1 if genome_a.fitness is None else genome_a.fitness + 1
            genome_b.fitness = -1 if genome_b.fitness is None else genome_b.fitness - 1
        else:
            genome_a.fitness = -1 if genome_a.fitness is None else genome_a.fitness - 1
            genome_b.fitness = 1 if genome_b.fitness is None else genome_b.fitness + 1
    # TODO: For now, since we're using total score, we don't need to bother with
    # this as fitness can't be negative.
    for (id, genome) in genomes:
        genome.fitness = max(0.001, genome.fitness)


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
    p.add_reporter(neat.Checkpointer(5, filename_prefix="knucklebones-"))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print(f'\nBest genome:\n{winner!s}')


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'knucklebones_config')
    run(config_path)
