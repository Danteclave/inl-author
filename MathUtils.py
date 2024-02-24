from typing import Callable, Sequence
import numpy as np


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


def group_blocks(blocks: Sequence[object], comparator: Callable[[object, object], bool]) -> Sequence[Sequence]:
    n = len(blocks)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if comparator(blocks[i], blocks[j]):
                uf.union(i, j)

    groups = []
    for i in range(n):
        root = uf.find(i)
        if root == i:
            groups.append([(blocks[j], j) for j in range(n) if uf.find(j) == root])

    return groups


def hamming_distance(list1: Sequence, list2: Sequence) -> int:
    a1 = np.array(list1)
    a2 = np.array(list2)
    return np.count_nonzero(np.bitwise_xor(a1, a2))
