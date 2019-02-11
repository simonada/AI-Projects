from testgraphs import SimpleGraph
from testgraphs import SimpleValuedGraph
import math
import random
from collections import defaultdict
import collections
import operator


def BFS(Graph, initialState):
    print("Starting search with BFS!")
    current_node = initialState
    path = []
    path.append(current_node)
    queue = collections.deque()
    visited = set()

    if Graph.isGoal(current_node):
        cost = len(path)
        return path, cost
    else:
        queue.append((path, Graph.successors(current_node)))
    while queue:
        # print('queue before pop', queue)
        predec, succ = queue.popleft()
        # print('checking new level')
        # print(predec, succ)
        for s in succ:
            if s not in visited:
                visited.add(s)
                # print('Next state: ', s)
                subpath = predec.copy()
                subpath.append(s)
                if Graph.isGoal(s):
                    cost = len(subpath)
                    print_solution(subpath, cost)
                    return subpath, cost
                else:
                    # idea is to be able to reconstruct the path through which the solution was found
                    # i.e. if the current successor did not lead to solution and we are going to visit him later once the level
                    # above is covered, we want to have in the queue also the path of how to reach this node
                    queue.append((subpath.copy(), Graph.successors(s)))
                    # we go to the next child, want to remove the sibling
                    subpath.clear()
            # else:
            #   print('Skipping node as it was already visited: ',s)
    print('Goal node not found.')
    return initialState, -1


def UCS(Graph, initialState):
    print("Starting search with UCS!")

    current_node = initialState
    path = []
    path.append(current_node)
    queue = collections.deque()
    visited = set()

    if Graph.isGoal(current_node):
        cost = evaluate_weighted_path(Graph, path, False)
        return path, cost
    else:
        queue.append((path, Graph.successors(current_node)))
    while queue:
        # print('queue before pop', queue)
        predec, succ = queue.popleft()
        succ.sort(key=lambda x: x[0])

        for cost, s in succ:
            # first visit all children on this level
            if s not in visited:
                visited.add(s)
                # print('checking successor node: ', s, ' with cost ', cost)
                subpath = predec.copy()
                subpath.append(s)

                if Graph.isGoal(s):
                    cost = evaluate_weighted_path(Graph, subpath, False)
                    print_solution(subpath, cost)
                    return subpath, cost
                else:
                    # print('subpath: ', subpath, 'cost: ', cost)
                    # will go down the level later, keep track of how we reached there.
                    queue.append((subpath.copy(), Graph.successors(s)))
                    subpath.clear()
            # else:
            #   print('Skipping node as it was already visited: ', s)

    print('Goal node not found.')
    return initialState, -1


def DFS(Graph, initialState):
    print("Starting search with DFS!")

    current_node = initialState
    path = []
    stack = collections.deque()
    stack.append((path, current_node))
    visited = set()

    while stack:
        # print('stack before pop', stack)
        predec, succ = stack.popleft()
        # print('stack', stack)
        if succ not in visited:
            visited.add(succ)
            # print('checking successor: ', succ)
            subpath = predec.copy()
            subpath.append(succ)
            if Graph.isGoal(succ):
                cost = len(subpath)
                print_solution(subpath, cost)
                return subpath, cost
            else:
                # print('subpath',subpath)
                for s in Graph.successors(succ):
                    # LIFO
                    stack.appendleft((subpath.copy(), s))
                subpath.clear()

        # else:
        #   print('Skipping node as it was already visited: ', succ)

    print('Goal node not found.')
    return initialState, -1


def DLS(Graph, initialState, depthLimit):
    print("Starting search with DLS with depth limit", depthLimit)

    path = collections.deque()
    path.append(initialState)
    result = recursiveDLS(Graph, initialState, depthLimit, path)
    print(result)
    if len(result) != 2:
        print("Found no solution due to", result, ".")
        return path, -1
    else:
        path, cost = result
        print_solution(path, cost)
        return path, cost


def recursiveDLS(Graph, current_node, depthLimit, path):
    cutoff = False
    # The root level is level = 0 !
    currentDepth = len(path) - 1
    #print('path ',path, ' depth ', currentDepth,' limit ',depthLimit)
    if Graph.isGoal(current_node):
        cost = len(path)
        return path, cost
    else:
        if currentDepth == depthLimit:
            #print('Cutoff occured.')
            return 'cutoff'
        for succ in Graph.successors(current_node):
            path.append(succ)
            result = recursiveDLS(Graph, succ, depthLimit, path)
            if result == 'cutoff':
                cutoff = True
                # the appended successor does not need to be kept in the path anymore
                path.pop()
            elif result != 'failure':
                return result
        if cutoff:
            return 'cutoff'
        else:
            return 'failure'


def IDS(Graph, initialState):
    print("Starting search with IDS!")

    path = collections.deque()
    path.append(initialState)
    depthLimit = 0
    while True:
        # print()
        # print('Checking for depth limit: ', depthLimit)
        result = recursiveDLS(Graph, initialState, depthLimit, path)
        if result != 'cutoff':
            if len(result) != 2:
                print("Found no solution due to", result, ".")
                return path, -1
            else:
                path, cost = result
                print_solution(path, cost)
                return path, cost
        depthLimit += 1


def Astar(ValuedGraph, initialState, heuristic):
    print("Starting search with A*!")

    # Idea:
    # first compute for each successor f(n) = g(n) + h(n)
    # g(n) = sum of costs over the edges leading to n
    # h(n) = estimated cost by heurisic (e.g. Manhattan)
    # then sort successors by f(n)
    # expand by min(f(n))

    current_node = initialState
    path = []
    path.append(current_node)
    queue = collections.deque()
    visited = set()

    if ValuedGraph.isGoal(current_node):
        cost = evaluate_weighted_path(ValuedGraph, path, False)
        return path, cost
    else:
        queue.append((path, ValuedGraph.successors(current_node)))
    while queue:
        predec, succ = queue.popleft()

        # print('succ before f(n) computation: ', succ)

        # BEFORE SORTING update edge cost to include the total estimated cost to the nodes
        for idx, ele in enumerate(succ):
            edgeCost, s = ele
            path_until_s = predec.copy()
            path_until_s.append(s)
            # print('predec', path_until_s)
            cost_until_n = evaluate_weighted_path(ValuedGraph, path_until_s, False)
            if heuristic.__name__ == "ucs_greedy_heuristic":
                cost_n_to_goal = heuristic(ValuedGraph, s)
            else:
                cost_n_to_goal = heuristic(s)
            # print('g(n): ', cost_until_n, ' h(n): ', cost_n_to_goal)
            estimated_total_cost = cost_until_n + cost_n_to_goal
            edgeCost = estimated_total_cost
            succ[idx] = (edgeCost, s)

        succ.sort(key=lambda x: x[0])

        # print('Successor states after f(n) computation and sorting: ', succ)

        for cost, s in succ:
            if s not in visited:
                visited.add(s)
                # print('checking successor node: ',s,' with cost ',cost)
                subpath = predec.copy()
                subpath.append(s)

                if ValuedGraph.isGoal(s):
                    cost = evaluate_weighted_path(ValuedGraph, subpath, False)
                    print_solution(subpath, cost)
                    return subpath, cost
                else:
                    # print('subpath: ',subpath, 'cost: ', cost)
                    queue.append((subpath.copy(), ValuedGraph.successors(s)))
                    subpath.clear()

    print('Goal node not found.')
    return initialState, -1


def ucs_greedy_heuristic(Graph, state):
    # Idea: return h(n), i.e. estimated distance to goal node
    return UCS(Graph, state)[1]


# heuristic applicable to nPuzzle
def manhattan_heuristic(state):
    target_coordinates = get_target_coord(state.value)
    current_coordinated = get_current_coord(state.value)
    manhattan_dist = 0
    for node in state.value:
        if node != 0:
            target_x, target_y = target_coordinates[node]
            current_x, current_y = current_coordinated[node]
            dx = abs(target_x - current_x)
            dy = abs(target_y - current_y)
            manhattan_dist += (dy + dx)
    return manhattan_dist


def get_current_coord(state):
    dict_coordinates = {}
    ind_x = 0
    ind_y = 0
    dimensions = int(math.sqrt(len(state)))
    # print(dimensions)
    count = 1
    for ele in state:
        if count % dimensions == 0:
            dict_coordinates[ele] = (ind_x, ind_y)
            ind_x += 1
            ind_y = 0
            count += 1
        else:
            dict_coordinates[ele] = (ind_x, ind_y)
            ind_y += 1
            count += 1
    return dict_coordinates


def get_target_coord(state):
    dict_target_coordinates = {}
    ind_x = 0
    ind_y = 0
    dimensions = int(math.sqrt(len(state)))
    # print(dimensions)
    count = 1
    for i in range(len(state) - 1):
        if count % dimensions == 0:
            dict_target_coordinates[count] = (ind_x, ind_y)
            ind_x += 1
            ind_y = 0
            count += 1
        else:
            dict_target_coordinates[count] = (ind_x, ind_y)
            ind_y += 1
            count += 1
    dict_target_coordinates[0] = (ind_x, ind_y)
    return dict_target_coordinates


def MCTS(ValuedGraph, state, budget):
    dict_visit_frq = defaultdict(int)
    dict_rewards = defaultdict(int)
    # successors is tuple (cost, node), want to keep only the succ nodes
    successors = [x[1] for x in ValuedGraph.successors(state)]

    # first the visit frq and rewards of all succ are 0
    for node in successors:
        dict_visit_frq[node] = 0
        dict_rewards[node] = 0

    for i in range(budget):
        # print()
        #print('Simulation', i)
        node_to_visit = tree_policy(state, successors, dict_visit_frq, dict_rewards)
        reward = default_policy(ValuedGraph, state, node_to_visit)

        # Update node information:
        dict_visit_frq[node_to_visit] = dict_visit_frq[node_to_visit] + 1
        dict_rewards[node_to_visit] = dict_rewards[node_to_visit] + reward

        # dict_visit_frq, dict_rewards = backup(path, delta, dict_visit_frq, dict_rewards)
        # print('visit frq states after simulation: ', dict_visit_frq)
        # print('rewards per node after simulation: ', dict_rewards)

    # interested only in succ nodes
    dict_visit_frq = {succ_node: dict_visit_frq[succ_node] for succ_node in successors}
    # return the node with highest frequency
    return max(dict_visit_frq.items(), key=operator.itemgetter(1))[0]


def tree_policy(state, successors, dict_visit_frq, dict_rewards):
    nr_times_state_visited = dict_visit_frq[state] + 1
    scores = []
    for s in successors:
        total_playout_reward = dict_rewards[s]
        nr_times_visited = dict_visit_frq[s]

        if nr_times_visited == 0:
            # print('Node has not been visited, so is selected for expansion by tree policy ', s)
            return s
        else:
            score = total_playout_reward / nr_times_visited
            + (1 / math.sqrt(2)) * math.sqrt(2 * math.log(nr_times_state_visited) / nr_times_visited)
            scores.append(score)

    # print('Scores for each successor in tree policy: ', scores)
    best_child = successors[scores.index(max(scores))]
    # print('Selected successor in Tree Policy: ', best_child)
    return best_child


def default_policy(ValuedGraph, state, node_to_visit):
    # include tha cost between the initial state and the considered successor as well

    actions = ValuedGraph.successors(node_to_visit)
    path = []
    path.append(state)
    path.append(node_to_visit)
    current_node = node_to_visit
    visited = list()
    visited.append(current_node)

    while not ValuedGraph.isGoal(current_node):
        current_node = random.choice(actions)[1]
        while current_node in visited:
            current_node = random.choice(actions)[1]
        # print('Random action chosen in default policy: ', current_node)
        path.append(current_node)
        visited.append(current_node)
        actions = ValuedGraph.successors(current_node)

    cost = evaluate_weighted_path(ValuedGraph, path, False)
    # Idea: the smaller the cost, the bigger the reward (cost being the distance)
    reward = 1 / cost
    # print('Cost and reward for the node: ', cost, reward)

    return round(reward, 3)


def iterative_MCTS_basic(graphV, initial_state, number_sumulations):
    print("Starting iterative MCTS for SimpleValuedGraph.")
    path = []
    path.append(initial_state)
    current_state = initial_state

    # interested to reach the end goal single node
    while not graphV.isGoal(path[-1]):
        next_state = MCTS(graphV, current_state, number_sumulations)
        path.append(next_state)
        current_state = next_state
    if graphV.isGoal(path[-1]):
        cost = len(path)
        print_solution(path, cost)
        return path, cost


def iterative_MCTS_npuzzle(graphV, initial_state, number_sumulations):
    print("Starting iterative MCTS for nPuzzle with budget", number_sumulations, ".")
    path = []
    path.append(initial_state)
    current_state = initial_state

    while not graphV.isGoal(current_state):
        next_state = MCTS(graphV, current_state, number_sumulations)
        #print('Checking next state: ', next_state)
        path.append(next_state)
        current_state = next_state
    if graphV.isGoal(path):
        cost = len(path)
        print_solution(path, cost)
        return path, cost


def backup(path, delta, dict_visit_frq, dict_rewards):
    print(path)
    for node in path:
        dict_visit_frq[node] = dict_visit_frq[node] + 1
        dict_rewards[node] = dict_rewards[node] + delta
    return dict_visit_frq, dict_rewards


def evaluate_simple_path(path):
    length = len(path)
    print()
    print('Path to end node: ', path)
    print('Cost of path: ', length)


def evaluate_weighted_path(GraphV, path, verbose=True):
    index = 1
    totalCost = 0
    # in the case of weighted graph, take edge cost until the current node
    if isinstance(GraphV, SimpleValuedGraph):
        for node in path:
            if index < len(path):
                succ = path[index]
                edgeCost = get_cost_to_n(GraphV.successors(node), succ)
                totalCost += edgeCost
                index += 1
            elif verbose:
                print()
                print('Path to end node: ', path)
                print('Cost of path: ', totalCost)
                return totalCost
    else:
        # otherwise the cost is how many inversions were needed to reach the state
        totalCost = len(path)
    return totalCost


def get_cost_to_n(successors, target):
    # print(successors, target)
    cost = [item for item in successors if item[1] == target]
    return cost[0][0]


# IDEA: prune the search space by not visiting successor states of unsolvable configurations!
def is_solvable(state):
    number_inversions = count_inversions(state.value)
    return (number_inversions % 2) == 0


def count_inversions(state):
    arr = state.copy()
    if 0 in state:
        arr.remove(0)
    n = len(arr)
    count = 0

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                count += 1

    return count

def print_solution(path, cost):
    print("List of states to end state:", path)
    print("Cost of the path:", cost)
    print()




if __name__ == '__main__':
    # DEMO on example graphs from the Project statement
    simple_graph = SimpleGraph()
    valued_graph = SimpleValuedGraph()

    print("BFS results:")
    BFS(simple_graph, 0)
    print("UCS results:")
    UCS(valued_graph, 0)
    print("DFS results:")
    DFS(simple_graph, 0)
    print("DLS results with depth limit 4:")
    DLS(simple_graph, 0, 4)
    print("IDS results:")
    IDS(simple_graph, 0)
    print("Astar results:")
    Astar(valued_graph, 0, ucs_greedy_heuristic)
    print("MCTS results:")
    # MCTS(valued_graph, 0, 5)
    iterative_MCTS_basic(valued_graph, 0, 5)
