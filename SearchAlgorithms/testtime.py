import random
import time
import signal
from testgraphs import nPuzzleGraph
from testgraphs import nPuzzleGraphValued
from testgraphs import State
import math
from search import BFS
from search import DFS
from search import UCS
from search import IDS
from search import DLS
from search import Astar
from search import manhattan_heuristic
from search import MCTS
from search import iterative_MCTS_npuzzle


def test_time(algorithm, limit):
    n = 3

    while (n <= limit):
        print('Testing algorithm: ', str(algorithm.__name__), ' for n = ', n)
        print()
        time_over_iterations = 0
        count_successfull_executions = 0
        cost_over_runs = []

        # 5 executions
        for i in range(1, 6):
            print("Iteration", i)
            if algorithm.__name__ == "Astar" or algorithm.__name__ == "iterative_MCTS_npuzzle" or algorithm.__name__ == "MCTS" or algorithm.__name__ == "UCS":
                puzzleGraph = nPuzzleGraphValued(n)
            else:
                puzzleGraph = nPuzzleGraph(n)
            initial_state, inversions = generate_random_state(n)
            print("Initial state : ", initial_state, ' with inversions: ', inversions)

            start = time.time()
            try:
                with timeout(seconds=1800):
                    if algorithm.__name__ == "MCTS" or algorithm.__name__ == "iterative_MCTS_npuzzle":
                        path, cost = algorithm(puzzleGraph, initial_state, n)
                    elif algorithm.__name__ == "Astar":
                        path, cost = algorithm(puzzleGraph, initial_state, manhattan_heuristic)
                    elif algorithm.__name__ == "DLS":
                        path, cost = algorithm(puzzleGraph, initial_state, 25)
                    else:
                        path, cost = algorithm(puzzleGraph, initial_state)
                    end = time.time()
                    cost_over_runs.append(cost)
                    elapsed_time = end - start
                    time_over_iterations += elapsed_time
                    count_successfull_executions += 1
                    print('Algorithm finished with cost', cost)
                    print_time_output(start, end)
                    print()
            except Exception as ex:
                if isinstance(ex, TimeoutError):
                    print('Algorithm took more than 30min. Interrupting...')
                    end = time.time()
                    print_time_output(start, end)
                else:
                    print("Error!", ex.with_traceback())
                # break

        if count_successfull_executions == 0:
            print('Algorithm took more than 30min to execute on all iterations.')
            print('AVG Execution time N/A.')
            return ('Maximum number of n reached: ', n)

        else:
            print("Total executions time for", count_successfull_executions, "executions:")
            print_time_output(0, time_over_iterations)
            avg_time = time_over_iterations / count_successfull_executions
            hours, rem = divmod(avg_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print('AVG Execution time for ', algorithm.__name__, ":")
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            print("Cost over runs and AVG:")
            print(cost_over_runs, sum(cost_over_runs) / len(cost_over_runs))
            print()
            # Reset values for next rounds of tests
            n += 2
            count_successfull_executions = 0
            time_over_iterations = 0


def generate_random_state(n):
    random_array = random.sample(range(0, n * n), n * n)
    number_inversions = count_inversions(random_array)
    # limit the number of inversions to reduce complexity, i.e. execution time
    while (number_inversions % 2) != 0 or (number_inversions > math.pow(n, 2) + n):
        random_array = make_even_inversions(random_array)
        number_inversions = count_inversions(random_array)

    return State(random_array), number_inversions


def make_even_inversions(state):
    arr = state.copy()
    n = len(arr)

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                return arr
    return arr


# code adapted from ref: https://www.geeksforgeeks.org/bubble-sort/
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


# ref: https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def print_time_output(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    #
    test_time(BFS, 3)
    # print()
    test_time(UCS, 3)
    # print()
    test_time(DFS, 3)
    print()
    #test_time(DLS, 3)
    print()
    #test_time(IDS, 3)
    print()
    test_time(Astar, 5)
    print()
    # test_time(MCTS, 3)
    # test_time(iterative_MCTS_npuzzle, 3)
