"""Exact strong-bisimulation quotient of a finite game graph with alternate turns.

This prototype uses an explicit graph and validates the group rule. It still
enumerates the graph to construct the partition. It is NOT an implicit proof
engine for chess regions. Its reduction does not predict a speedup for eight
pieces.
"""
from collections import deque
import numpy as np


def sparse_oracle(successors, seeds):
    """WDL + exact distance to a decisive terminal for nonzero seeds at depth 0."""
    n = len(successors)
    parents = [[] for _ in range(n)]
    for i, children in enumerate(successors):
        for j in children: parents[j].append(i)
    state = np.asarray(seeds, dtype=np.int8).copy()
    depth = np.full(n, -1, np.int64); depth[state != 0] = 0
    remaining = np.asarray([len(x) for x in successors], np.int64)
    latest = np.zeros(n, np.int64)
    queue = deque(map(int, np.flatnonzero(state)))
    while queue:
        child = queue.popleft()
        for parent in parents[child]:
            if state[parent]: continue
            if state[child] == -1:
                state[parent] = 1
                depth[parent] = depth[child] + 1
                queue.append(parent)
            else:
                latest[parent] = max(latest[parent], depth[child])
                remaining[parent] -= 1
                if remaining[parent] == 0:
                    state[parent] = -1
                    depth[parent] = latest[parent] + 1
                    queue.append(parent)
    return state, depth


def refine(successors, seeds, progress=None):
    """Merge only nodes with equal primitive labels and successor-block sets."""
    seeds = np.asarray(seeds, np.int8)
    labels = [int(s) if s else (0 if not successors[i] else 2) for i, s in enumerate(seeds)]
    initial = {}; colors = []
    for label in labels:
        if label not in initial: initial[label] = len(initial)
        colors.append(initial[label])
    old_count = len(initial)
    rounds = 0
    while True:
        signatures = {}; new = []
        for i, children in enumerate(successors):
            signature = (colors[i], tuple(sorted({colors[j] for j in children})))
            if signature not in signatures: signatures[signature] = len(signatures)
            new.append(signatures[signature])
        rounds += 1
        count = len(signatures)
        if progress: progress(rounds, count)
        colors = new
        if count == old_count: break
        old_count = count
    blocks = [[] for _ in range(count)]
    for i, block in enumerate(colors): blocks[block].append(i)
    quotient = [sorted({colors[j] for j in successors[block[0]]}) for block in blocks]
    qseeds = [int(seeds[block[0]]) for block in blocks]
    # A separate full check. Outcome samples alone do not decide the groups.
    for k, block in enumerate(blocks):
        reference = tuple(quotient[k])
        for i in block:
            assert labels[i] == labels[block[0]]
            assert tuple(sorted({colors[j] for j in successors[i]})) == reference
    return quotient, qseeds, np.asarray(colors, np.int64), rounds


def solve_quotient(successors, seeds, progress=None):
    quotient, qseeds, colors, rounds = refine(successors, seeds, progress)
    qw, qd = sparse_oracle(quotient, qseeds)
    return qw[colors], qd[colors], dict(concrete_nodes=len(successors),
        concrete_edges=sum(map(len, successors)), quotient_nodes=len(quotient),
        quotient_edges=sum(map(len, quotient)), refinement_rounds=rounds,
        node_reduction=len(successors)/len(quotient))
