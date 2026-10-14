"""Exact WDL research kernels. This is not a production generator of chess tablebases.

States use the view of the player to move: -1 loss, 0 unresolved/draw, 1 win.
Terminal losses and wins are trusted seeds. Terminal draws have no edges out
and stay at zero. Every array must describe a complete legal successor
relation. Represent each external dependency in that relation as a seed.
"""
from collections import deque
import random
import numpy as np


def counter_oracle(adjacency, seeds):
    """Independent retrograde reference with predecessor counters, linear in edges."""
    a = np.asarray(adjacency, dtype=bool)
    state = np.asarray(seeds, dtype=np.int8).copy()
    remaining = a.sum(axis=1).astype(np.int64)
    parents = [np.flatnonzero(a[:, j]) for j in range(len(state))]
    queue = deque(map(int, np.flatnonzero(state)))
    while queue:
        child = queue.popleft()
        for parent in parents[child]:
            if state[parent]:
                continue
            if state[child] == -1:
                state[parent] = 1
                queue.append(int(parent))
            else:
                remaining[parent] -= 1
                if remaining[parent] == 0:
                    state[parent] = -1
                    queue.append(int(parent))
    return state


def encoded_float_predicates(adjacency, state, block=64):
    """FP16 operand / FP32 accumulator emulation of bounded exact reductions.

    `z = 65 * nonwin + loss`. Here `nonwin` counts the children that are not
    a win. `loss` counts the children that are a loss.
    Each reduction has at most 64 children. `z % 65` finds a child that is a
    loss. `z == 0` means that every child in that block is a win. Never add z
    values across blocks before you decode them. The low digit would overflow
    its radix. NumPy casts the FP16 operands back to FP32 for the
    multiplication. Thus this validates the arithmetic, not the CUDA execution
    or speed of tensor cores.
    """
    if not 1 <= block <= 64:
        raise ValueError("block must be between 1 and 64")
    a = np.asarray(adjacency, dtype=bool)
    state = np.asarray(state, dtype=np.int8)
    if a.shape != (len(state), len(state)):
        raise ValueError("adjacency must be square and match state")
    if not np.isin(state, [-1, 0, 1]).all():
        raise ValueError("invalid WDL state")
    win = np.zeros(len(state), dtype=bool)
    bad = np.zeros(len(state), dtype=bool)
    encoded = (65 * (state != 1) + (state == -1)).astype(np.float16)
    for start in range(0, len(state), block):
        stop = min(start + block, len(state))
        z = a[:, start:stop].astype(np.float16).astype(np.float32) @ encoded[start:stop].astype(np.float32)
        if not np.equal(z, np.rint(z)).all() or np.max(z, initial=0) > 4224:
            raise AssertionError("bounded exact-integer accumulation violated")
        counts = z.astype(np.int32)
        win |= counts % 65 != 0
        bad |= counts != 0
    loss = (a.sum(axis=1) > 0) & ~bad
    return win, loss


def float_fixed_point(adjacency, seeds):
    state = np.asarray(seeds, dtype=np.int8).copy()
    rank = np.full(len(state), -1, dtype=np.int64)
    rank[state != 0] = 0
    rounds = 0
    while True:
        win, loss = encoded_float_predicates(adjacency, state)
        unresolved = state == 0
        new = state.copy()
        new[unresolved & win] = 1
        new[unresolved & ~win & loss] = -1
        changed = new != state
        if not changed.any():
            return state, rank, rounds
        rounds += 1
        rank[changed] = rounds
        state = new


def sharded_dirty_solver(adjacency, seeds, shards=4, rng_seed=0):
    """Asynchronous owner/dirty simulation with delayed duplicate notifications.

    Values are globally visible in this CPU model. A production implementation
    must route proven values or necessary boundary snapshots with notifications.
    Notifications are idempotent dirty marks, not counter decrements.
    """
    a = np.asarray(adjacency, dtype=bool)
    state = np.asarray(seeds, dtype=np.int8).copy()
    parents = [np.flatnonzero(a[:, j]).tolist() for j in range(len(state))]
    dirty = [set(range(i, len(state), shards)) for i in range(shards)]
    messages = []
    rng = random.Random(rng_seed)
    updates = deliveries = duplicates = 0
    while any(dirty) or messages:
        if messages and (not any(dirty) or rng.random() < .5):
            j = rng.randrange(len(messages))
            owner, node = messages.pop(j)
            dirty[owner].add(node)
            deliveries += 1
            continue
        owners = [i for i, work in enumerate(dirty) if work]
        owner = rng.choice(owners)
        node = dirty[owner].pop()
        if state[node]:
            continue
        child_states = state[np.flatnonzero(a[node])]
        if np.any(child_states == -1):
            value = 1
        elif len(child_states) and np.all(child_states == 1):
            value = -1
        else:
            continue
        state[node] = value
        updates += 1
        for parent in parents[node]:
            messages.append((parent % shards, parent))
            if rng.random() < .25:
                messages.append((parent % shards, parent))
                duplicates += 1
    return state, dict(updates=updates, notifications=deliveries, duplicates=duplicates)


def verify_certificate(adjacency, seeds, state, rank):
    """Reject self-consistent labels without support. Finite ranks must decrease."""
    a = np.asarray(adjacency, dtype=bool)
    for i, value in enumerate(state):
        children = np.flatnonzero(a[i])
        if seeds[i]:
            assert value == seeds[i] and rank[i] == 0
        elif value == 1:
            assert any(state[j] == -1 and 0 <= rank[j] < rank[i] for j in children)
        elif value == -1:
            assert len(children) and all(state[j] == 1 and 0 <= rank[j] < rank[i] for j in children)
        else:
            assert not any(state[j] == -1 for j in children)
            assert not len(children) or any(state[j] == 0 for j in children)
    return True


def segmented_slider_predicates(occupied, safe_destination, valid_source, state, kind):
    """Factorized operator for quiet slider moves, without a 64x64 relation.

    The non-king piece that moves is absent from occupied. Other pieces are
    fixed. A legal chess implementation supplies the source validity and the
    own-king safety of each destination. This operator does not include
    captures or the moves of other pieces. It uses one pair of counts again
    for all squares in each unblocked line segment.
    """
    modes = {'rook': (0, 1), 'bishop': (2, 3), 'queen': (0, 1, 2, 3)}[kind]
    occupied = set(occupied)
    safe_destination = np.asarray(safe_destination, bool)
    valid_source = np.asarray(valid_source, bool)
    state = np.asarray(state, np.int8)
    win = np.zeros(64, bool)
    bad = np.zeros(64, bool)
    degree = np.zeros(64, np.int16)
    for mode in modes:
        lines = {}
        for sq in range(64):
            f, r = sq % 8, sq // 8
            key = r if mode == 0 else f if mode == 1 else f-r if mode == 2 else f+r
            lines.setdefault(key, []).append(sq)
        for line in lines.values():
            segment = []
            for sq in line + [None]:
                if sq is not None and sq not in occupied:
                    segment.append(sq)
                    continue
                if segment:
                    cells = np.asarray(segment, np.intp)
                    safe = safe_destination[cells]
                    losses = safe & (state[cells] == -1)
                    nonwins = safe & (state[cells] != 1)
                    win[cells] |= losses.sum() - losses.astype(np.int16) > 0
                    bad[cells] |= nonwins.sum() - nonwins.astype(np.int16) > 0
                    degree[cells] += safe.sum() - safe.astype(np.int16)
                segment = []
    return valid_source & win, valid_source & (degree > 0) & ~bad
