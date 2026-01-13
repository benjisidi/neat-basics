from collections import Counter, defaultdict, abc
import os
import neat
import numpy as np
from itertools import combinations
from tqdm import tqdm
import math
import pyinputplus as pyip
import multiprocessing


def get_net_input(board1: np.ndarray, board2: np.ndarray, roll: int) -> np.ndarray:
    # We've got:
    # - 9x7-length vectors for each board space, one-hot representing empty -> 6
    # - Another of the above for opponent board
    # - 1x6 for one-hot encoding of current roll
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
                       turn_counter: int, opponent_index: int, roll: int, interactive=False) -> list[np.ndarray]:
    # First, find the most preferred legal move by iterating through the preferences til a legal one is found
    player_board = boards[turn_counter]
    player_cols = np.array_split(player_board, 3)
    desired_move = -1
    # Argsort is always descending, so we'll reverse the arry to get the most preferred moves first
    move_preference_indices = np.argsort(move_preferences)[::-1]
    # We're going to check each col in order of preference, finding the first one with an empty space
    for col in move_preference_indices:
        if np.any(player_cols[col] == 0):
            # This column definitely has a space, so we'll play in the first one we find
            column_indices = np.array([0, 1, 2]) + 3 * col
            for i in column_indices:
                if player_board[i] == 0:
                    desired_move = i
            break
    if desired_move == -1:
        raise ValueError(
            "No valid moves. Are you trying to play when the game is over?")
    if interactive:
        print(f"Opponent places {roll} in slot {desired_move}")
    return process_move(boards, desired_move, turn_counter, opponent_index, roll)


def process_move(boards: list[np.ndarray], move: int,
                 turn_counter: int, opponent_index: int, roll: int) -> list[np.ndarray]:
    player_board = boards[turn_counter]
   # Place the roll in the empty space of the current player's board
    player_board[move] = roll
    # Last, blank any matching rolls from the opposing player's column
    opponent_board = boards[opponent_index]
    column = move // 3
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


def get_random_move(boards: list[np.ndarray], turn_counter: int) -> int:
    board = boards[turn_counter]
    return np.random.choice(np.argwhere(board == 0).flatten())


def net_vs_func(net: neat.nn.FeedForwardNetwork, func: abc.Callable[[list[np.ndarray], int], int]) -> list[int]:
    """
    Net plays an arbitrary hardcoded function for one game.
    If net wins, returns 1
    Else returns 0
    """
    turn_counter = np.random.randint(0, 2)
    game_over = False
    turns = 0
    max_turns = 200
    boards = [get_blank_board(), get_blank_board()]
    while not game_over and turns < max_turns:
        roll = np.random.randint(1, 7)
        # We'll hardcode the net to be player 0 and func to be player 1
        # First play is randomized so this is arbitrary
        if turn_counter == 0:
            opponent_index = 1
            move_preferences = net.activate(get_net_input(
                boards[turn_counter], boards[opponent_index], roll))
            boards = process_net_output(
                boards, move_preferences, turn_counter, opponent_index, roll)
        else:
            opponent_index = 0
            move = func(boards, turn_counter)
            boards = process_move(
                boards, move, turn_counter, opponent_index, roll)

        game_over = is_game_over(boards)
        turn_counter = opponent_index
        turns += 1
    scores = get_scores(boards)
    return scores


def net_vs_net(net_a: neat.nn.FeedForwardNetwork, net_b: neat.nn.FeedForwardNetwork):
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
    nets = {}
    for id, genome in genomes:
        nets[id] = neat.nn.FeedForwardNetwork.create(genome, config)

    for pairing in tqdm(combinations(genomes, 2), total=math.comb(len(genomes), 2)):
        [(id_a, genome_a), (id_b, genome_b)] = pairing
        net_a = nets[id_a]
        net_b = nets[id_b]
        score_a, score_b = net_vs_net(net_a, net_b)
        if score_a > score_b:
            genome_a.fitness = (genome_a.fitness or 0) + 1
            genome_b.fitness = (genome_b.fitness or 0) - 1
        else:
            genome_a.fitness = (genome_a.fitness or 0) - 1
            genome_b.fitness = (genome_b.fitness or 0) + 1
    for (id, genome) in genomes:
        genome.fitness = max(0.001, genome.fitness)


def eval_genome_vs_func(genome, config, func: abc.Callable[[list[np.ndarray], int], int]):
    n_games = 500
    total_score_diff = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for _ in range(n_games):
        scores = net_vs_func(net, func)
        total_score_diff += scores[0] - scores[1]
    return max(total_score_diff/n_games, 0.001)


def eval_genome_vs_random(genome, config):
    return eval_genome_vs_func(genome, config, get_random_move)


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
    p.add_reporter(neat.Checkpointer(
        5, filename_prefix="checkpoint-knucklebones-"))

    # Run for up to 300 generations.
    with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome_vs_random) as evaluator:
        winner = p.run(evaluator.evaluate, 300)

    # Display the winning genome.
    print(f'\nBest genome:\n{winner!s}')


def print_boards(board1, board2):
    print(board2.reshape(3, 3).T)
    print("--------")
    print(board1.reshape(3, 3).T)


def play_game_interactive(net: neat.nn.FeedForwardNetwork):
    boards = [get_blank_board(), get_blank_board()]
    turn_counter = np.random.randint(0, 2)
    game_over = False
    turns = 0
    while not game_over:
        opponent_index = (turn_counter + 1) % 2
        if turn_counter == 0:
            roll = np.random.randint(1, 7)
            move_preferences = net.activate(get_net_input(
                boards[turn_counter], boards[opponent_index], roll))
            print("Opponent rolls", roll)
            boards = process_net_output(
                boards, move_preferences, turn_counter, opponent_index, roll, interactive=True)
        else:
            roll = np.random.randint(1, 7)
            print("Your roll is", roll)
            print_boards(boards[turn_counter], boards[opponent_index])
            move = pyip.inputNum("Enter move (1-9):") - 1
            boards = process_move(
                boards, move, turn_counter, opponent_index, roll)
        scores = get_scores(boards)
        print(f"Scores: You {int(scores[1])} | Bot {int(scores[0])}")
        print("\n")
        game_over = is_game_over(boards)
        turn_counter = opponent_index
        turns += 1

    scores = get_scores(boards)
    return scores


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    interactive = False
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'knucklebones_config')
    if interactive:
        pop = neat.Checkpointer.restore_checkpoint("checkpoint-knucklebones-8")
        (gid, best_genome) = list(pop.population.items())[0]
        print(gid, best_genome.fitness)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        play_game_interactive(net)
    else:
        run(config_path)


# TODO: The bots need a constant opponent to be evaluated against, that way
# better bot = higher fitness consistently.
# Ideas for opponents are random player, and greedy player.
# Ideas for fitness score could be games won out of 50/100 whatever, or total
# score diff or similar
