import random
import math
import numpy as np
from collections import defaultdict
import math
import random
import operator
from Clickomania import Clickomania


def clickomaniaPlayer(Clickomania, init_state):
    # --- HYPERPARAMETERS ---

    # available UCS, MCTS
    player_strategy = "MCTS"
    MCTS_simulations = 10

    # --- END HYPERPARAMETERS ---

    initial_state = init_state
    game = Clickomania
    print("New Game started! Initial state:")
    print_state(initial_state, Clickomania)
    Clickomania.set_init_state(init_state)
    game_score = 0

    if player_strategy == "UCS":
        state_sequence, game_score = UCS(Clickomania, init_state)
    if player_strategy == "MCTS":
        state_sequence, game_score = iterative_MCTS(Clickomania, init_state, MCTS_simulations)

    if (game_score < 0):
        game_score = 0

    print("Final Score: ", game_score)
    print("Number of States to End State: ", len(state_sequence))

    return state_sequence, game_score


def print_state(state, game):
    game_array = np.array(state)
    game_matrix = game_array.reshape(game.N, game.M)
    print(game_matrix)


def compute_penalty(state):
    total_reduction = 0
    K = len(set(state))
    for color in range(1, K + 1):
        indices = [i for i, x in enumerate(state) if x == color]
        if indices:
            remaining_tiles = len(indices)
            reduction = int(math.pow((remaining_tiles - 1), 2))
            total_reduction += reduction
    return total_reduction


def UCS(Clickomania, initialState):
    print()
    print("UCS playing...")
    current_node = initialState
    path = []
    path.append(current_node)
    queue = list()
    game_score = 0

    if Clickomania.isGoal(current_node):
        return current_node, 0
    else:
        queue = Clickomania.successors(current_node)
    while queue:
        queue.sort(key=lambda x: x[0], reverse=True)
        reward, state = queue.pop(0)

        if Clickomania.isGoal(state):
            print("Final state with reward", reward, ":")
            path.append(state)
            game_score += reward
            print_state(state, Clickomania)
            print("Game over!")
            game_score -= compute_penalty(state)
            return path, game_score

        if state not in path:
            path.append(state)
            print('Next state with reward', reward, ":")
            print_state(state, Clickomania)
            game_score += reward
            queue = Clickomania.successors(state)
    return path, game_score


def iterative_MCTS(graphV, initial_state, number_sumulations):
    print()
    print("MCTS playing...")
    path = []
    path.append(initial_state)
    current_state = initial_state
    total_reward = 0
    # interested to reach the end goal single node
    while not graphV.isGoal(path[-1]):
        next_state, reward = MCTS(graphV, current_state, number_sumulations)
        total_reward += reward
        path.append(next_state)
        current_state = next_state
    if graphV.isGoal(path[-1]):
        total_reward -= compute_penalty(path[-1])
        print("Game over!")
        return path, total_reward


def MCTS(Clickomania, state, budget):
    dict_visit_frq = defaultdict(int)
    dict_accumulated_rewards = defaultdict(int)
    dict_single_move_reward = defaultdict(int)
    # successors is tuple (reward, node), want to keep only the succ nodes
    successors = [x[1] for x in Clickomania.successors(state)]

    # first the visit frq and rewards of all succ are 0
    for reward, node in Clickomania.successors(state):
        dict_visit_frq[repr(node)] = 0
        dict_accumulated_rewards[repr(node)] = 0
        dict_single_move_reward[repr(node)] = reward

    for i in range(budget):
        # print()
        # print('New Simulation!')
        node_to_visit = tree_policy(state, successors, dict_visit_frq, dict_accumulated_rewards)
        reward = default_policy(Clickomania, state, node_to_visit)

        # Update node information:
        dict_visit_frq[repr(node_to_visit)] = dict_visit_frq[repr(node_to_visit)] + 1
        dict_accumulated_rewards[repr(node_to_visit)] = dict_accumulated_rewards[repr(node_to_visit)] + reward

        # dict_visit_frq, dict_accumulated_rewards = backup(path, delta, dict_visit_frq, dict_accumulated_rewards)
    # print('visit frq states after simulation: ', dict_visit_frq)
    # print('rewards per node after simulation: ', dict_accumulated_rewards)

    # interested only in succ nodes
    # dict_visit_frq = { succ_node: dict_visit_frq[succ_node] for succ_node in successors }
    # return the node with highest frequency
    # print(dict_visit_frq)

    best_state = eval(max(dict_visit_frq.items(), key=operator.itemgetter(1))[0])
    single_move_reward = dict_single_move_reward[repr(best_state)]
    print('Next state with reward', single_move_reward, ":")
    print_state(best_state, Clickomania)

    return best_state, single_move_reward


def tree_policy(state, successors, dict_visit_frq, dict_accumulated_rewards):
    exploration_factor_C = 10

    # print('Tree Policy')
    nr_times_parent_visited = dict_visit_frq[repr(state)] + 1
    scores = []
    for s in successors:
        total_playout_reward = dict_accumulated_rewards[repr(s)]
        nr_times_visited = dict_visit_frq[repr(s)]

        if nr_times_visited == 0:
            # print('Node has not been visited, so is selected for expansion by tree policy ', s)
            # print('Selected successor in Tree Policy: ', s)
            return s
        else:
            score = total_playout_reward / nr_times_visited
            + exploration_factor_C * math.sqrt(2 * math.log(nr_times_parent_visited) / nr_times_visited)
            scores.append(score)

    # print('Scores for each successor in tree policy: ', scores)
    best_child = successors[scores.index(max(scores))]
    # print('Selected successor in Tree Policy: ', best_child)
    return best_child


def default_policy(Clickomania, parent_state, node_to_visit):
    # Here the Rollout happens, i.e. from the chosen state actions are chosen randomly until the end state is reached

    # print('Default policy.')
    actions = Clickomania.successors(node_to_visit)
    current_node = node_to_visit
    visited = list()
    visited.append(current_node)
    total_reward = 0

    while not Clickomania.isGoal(current_node):
        # action = (reward, next state)
        reward, current_node = random.choice(actions)
        while current_node in visited:
            current_node = random.choice(actions)[1]
        # print('Random action chosen in default policy: ', current_node)
        visited.append(current_node)
        actions = Clickomania.successors(current_node)
        total_reward += reward

    total_reward -= compute_penalty(current_node)

    # print('reward and reward for the node: ',reward, reward)

    return total_reward


def print_performance_statistics(performance_array):
    print("MAX:", max(performance_array))
    print("MIN:", min(performance_array))
    print("AVG:", sum(performance_array) / len(performance_array))


if __name__ == '__main__':
    M = 5
    N = 5
    K = 3

    sequences_MCTS = []
    scores_MCTS = []

    sequences_UCS = []
    scores_UCS = []

    game = Clickomania(N, M, K)
    #good_test_state = [2, 3, 2, 2, 3, 1, 1, 2, 3, 3, 3, 3]
    initial_state = game.random_init_state()

    for i in range(3):
        #initial_state = game.random_init_state()

        # One can adjust the clickomaniaPlayer function to take as additional parameter the name of the algorithm to use.

        #state_sequence, game_score = clickomaniaPlayer(game, initial_state, "UCS")
        #sequences_UCS.append(len(state_sequence))
        #scores_UCS.append(game_score)
        #print()
        state_sequence, game_score = clickomaniaPlayer(game, initial_state)
        sequences_MCTS.append(len(state_sequence))
        scores_MCTS.append(game_score)
        print()

    print("MCTS Statistics (max),(min),(avg)")
    print("Game Length:")
    print_performance_statistics(sequences_MCTS)
    print("Game Scores:")
    print_performance_statistics(scores_MCTS)
    print()
    # Used to make the comparison as described in the paper.
    #print("UCTS Statistics (max),(min),(avg)")
    #print("Game Length:")
    #print_performance_statistics(sequences_UCS)
    #print("Game Scores:")
    #print_performance_statistics(scores_UCS)
