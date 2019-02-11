import random
import math
import numpy as np
from collections import defaultdict
import math
import random
import operator


class Clickomania:

    def __init__(self, N, M, K):
        self.N = N
        self.M = M
        self.K = K
        self.initial_state = self.random_init_state()

    def random_init_state(self):
        number_tiles = self.M * self.N
        start = 1
        number_colors = self.K
        res = []
        for j in range(number_tiles):
            res.append(random.randint(start, number_colors))

        return res

    def set_init_state(self, init_state):
        self.initial_state = init_state
        # print(self.initial_state)

    def successors(self, state):
        succs = []
        all_groups = []
        for color in range(1, self.K + 1):
            indices = [i for i, x in enumerate(state) if x == color]
            grouped_indices = self.group_adjacent_tiles(indices)
            for group in grouped_indices:
                # print("group",group)
                all_groups.append(group)

        for tiles_group in all_groups:
            state_gain = self.compute_state_gain(tiles_group)
            succ_state = self.update_state(state, tiles_group)
            succs.append((state_gain, succ_state))
        return succs

    def group_adjacent_tiles(self, indices):
        same_color_tiles = indices.copy()
        index_groups = []

        for ind in same_color_tiles:
            group = []
            group.append(ind)
            neighbours_to_expand = set(self.get_adjacent_tiles(ind))
            # all_adjacent = get_adjacent_tiles(ind)

            while neighbours_to_expand:
                # print("queue",neighbours_to_expand)
                adj = neighbours_to_expand.pop()
                if adj in same_color_tiles:
                    if adj not in group:
                        same_color_tiles.remove(adj)
                        group.append(adj)
                        neighbours_to_expand.update(self.get_adjacent_tiles(adj))
            if len(group) >= 2:
                index_groups.append(group)
        return index_groups

    # Idea: find the indices of all block that potentially could build a block with the input index
    def get_adjacent_tiles(self, ind):
        adj = []
        M = self.M
        N = self.N
        # LEFT
        if (ind % M != 0):
            adj.append(ind - 1)
        # UP
        if (ind >= M):
            adj.append(ind - M)
        # RIGHT
        if (ind % M != M - 1):
            adj.append(ind + 1)
        # DOWN
        if (ind < M * (M - 1) - M):
            adj.append(ind + M)
        return adj

    def compute_state_gain(self, tiles_group):
        removed_tiles = len(tiles_group)
        gain = math.pow(removed_tiles - 1, 2)
        return int(gain)

    def update_state(self, state, tiles_group):
        next_state = state.copy()
        M = self.M
        # remove all blocks belonging to the group, i.e. set index of the position to zero
        # The blocks should be replaced by the upper level numbers if possible
        for ind in tiles_group:
            if ind >= M:
                color_tile_above = next_state[ind - M]
                next_state[ind] = color_tile_above
                next_state[ind - M] = 0
            else:
                next_state[ind] = 0
        # Check for empty columns
        next_state = self.shift_empty_columns(next_state)
        next_state = self.shift_empty_rows(next_state)

        return next_state

    def shift_empty_columns(self, state):
        next_state = state.copy()
        M = self.M
        N = self.N
        for i in range(0, M - 1):
            column_colors = []
            for j in range(0, N):
                # e.g. if we're at column 0, then we want colors of 0, 4, 8 from the array
                column_colors.append(next_state[i + M * (j)])
            check_empty = all(v == 0 for v in column_colors)
            if check_empty:
                # print('shifting empty column')
                for j in range(0, N):
                    # replace all values in the column with the values of the column to the right
                    next_state[i + M * (j)] = next_state[i + M * (j) + 1]
                    # set the column to the right to zero
                    next_state[i + M * (j) + 1] = 0
        return next_state

    def shift_empty_rows(self, state):
        next_state = state.copy()
        M = self.M
        N = self.N
        # don't care if most upper row is empty
        # go bottom-up
        for i in range(N, 1, -1):
            row_colors = []
            for j in range(0, M):
                if next_state[i * M - j - 1] == 0:
                    if next_state[i * M - j - M - 1] != 0:
                        next_state[i * M - j - 1] = next_state[i * M - j - M - 1]
                        next_state[i * M - j - M - 1] = 0
        return next_state

    def isGoal(self, state):
        succ = self.successors(state)
        # if empty = no tiles could be grouped
        if not succ:
            return True
        return False
