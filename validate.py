"""Run independent correctness checks. CPU only, with no claims about H100 times."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from reference import counter_oracle, float_fixed_point, sharded_dirty_solver, verify_certificate, encoded_float_predicates, segmented_slider_predicates


def random_graphs(cases=120):
    rng = np.random.default_rng(8261)
    sizes = [17, 63, 64, 65, 127, 128, 129, 255]
    stats = dict(cases=cases, nodes=0, duplicate_notifications=0, max_rounds=0)
    for c in range(cases):
        n = sizes[c % len(sizes)]
        a = rng.random((n, n)) < rng.uniform(.004, .12)
        seeds = np.zeros(n, dtype=np.int8)
        terminals = rng.choice(n, max(2, n // 10), replace=False)
        a[terminals] = False
        seeds[terminals] = rng.choice([-1, 0, 1], len(terminals))
        expected = counter_oracle(a, seeds)
        got, rank, rounds = float_fixed_point(a, seeds)
        assert np.array_equal(got, expected), f"floating graph mismatch {c}"
        assert verify_certificate(a, seeds, got, rank)
        for shards in (1, 2, 7):
            actual, metrics = sharded_dirty_solver(a, seeds, shards, c)
            assert np.array_equal(actual, expected), f"sharding mismatch {c}/{shards}"
            stats['duplicate_notifications'] += metrics['duplicates']
        stats['nodes'] += n
        stats['max_rounds'] = max(stats['max_rounds'], rounds)
    # A long forced chain tests proof depth and parity. A closed cycle is a draw.
    n = 257
    a = np.zeros((n, n), bool)
    for i in range(n - 1): a[i, i + 1] = True
    seeds = np.zeros(n, np.int8); seeds[-1] = -1
    got, rank, rounds = float_fixed_point(a, seeds)
    assert rounds == 256 and rank[0] == 256
    assert np.array_equal(got, counter_oracle(a, seeds))
    assert verify_certificate(a, seeds, got, rank)
    # A radix carry across blocks would incorrectly hide children that are a loss.
    a = np.ones((130, 130), bool); state = np.full(130, -1, np.int8)
    w, l = encoded_float_predicates(a, state)
    assert w.all() and not l.any()
    stats['chain_depth'] = 256
    stats['radix_carry_regression'] = 'pass'
    return stats


def chess_fibers(chess_path, cases=160):
    sys.path.insert(0, str(Path(chess_path).resolve()))
    import chess
    rng = np.random.default_rng(9383)
    checked_rows = checked_edges = safe_segment_rows = 0
    for case in range(cases):
        piece = [chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KNIGHT][case % 4]
        # Skeleton: the kings and 2 or 3 other pieces are fixed.
        # The square of one white piece is variable.
        while True:
            squares = rng.choice(64, 5, replace=False).tolist()
            if chess.square_distance(squares[0], squares[1]) > 1: break
        board = chess.Board(None)
        fixed = [(squares[0], chess.KING, chess.WHITE), (squares[1], chess.KING, chess.BLACK),
                 (squares[2], chess.ROOK, chess.BLACK), (squares[3], chess.KNIGHT, chess.WHITE),
                 (squares[4], chess.BISHOP, chess.BLACK)]
        for sq, kind, color in fixed: board.set_piece_at(sq, chess.Piece(kind, color))
        board.turn = chess.WHITE
        a = np.zeros((64, 64), bool)
        valid = np.zeros(64, bool)
        for x in range(64):
            if board.piece_at(x): continue
            board.set_piece_at(x, chess.Piece(piece, chess.WHITE))
            if board.is_valid():
                valid[x] = True
                for move in board.legal_moves:
                    if move.from_square == x and not board.is_capture(move):
                        a[x, move.to_square] = True
            board.remove_piece_at(x)
        state = rng.choice([-1, 0, 1], 64).astype(np.int8)
        w, l = encoded_float_predicates(a, state)
        assert np.array_equal(w, (a & (state == -1)[None, :]).any(1))
        assert np.array_equal(l, a.any(1) & ~(a & (state != 1)[None, :]).any(1))
        checked_rows += int(valid.sum()); checked_edges += int(a.sum())
        # Factorized quiet relation for a non-king piece. It is the geometry from
        # fixed blockers, times a mask of own-king safety for each destination,
        # times source validity.
        safe_destination = np.zeros(64, bool)
        for y in range(64):
            if board.piece_at(y): continue
            board.set_piece_at(y, chess.Piece(piece, chess.WHITE))
            safe_destination[y] = not board.is_check()
            board.remove_piece_at(y)
        geometric = np.zeros((64, 64), bool)
        for x in range(64):
            if board.piece_at(x): continue
            board.set_piece_at(x, chess.Piece(piece, chess.WHITE))
            destinations = board.attacks(x) & ~board.occupied
            for y in destinations: geometric[x, y] = True
            board.remove_piece_at(x)
        factored = valid[:, None] & geometric & safe_destination[None, :]
        assert np.array_equal(factored, a), f"fiber legality factorization mismatch {case}"
        # For sliders, an edge connects every unblocked pair on the same line
        # segment. This checks the segment relation against legal moves.
        if piece != chess.KNIGHT:
            kinds = ([0, 1] if piece == chess.ROOK else [2, 3] if piece == chess.BISHOP else [0, 1, 2, 3])
            segment_relation = np.zeros_like(a)
            occupied = {sq for sq, _, _ in fixed}
            for kind in kinds:
                lines = {}
                for sq in range(64):
                    f, r = chess.square_file(sq), chess.square_rank(sq)
                    key = r if kind == 0 else f if kind == 1 else f-r if kind == 2 else f+r
                    lines.setdefault(key, []).append(sq)
                for line in lines.values():
                    segment = []
                    for sq in line + [None]:
                        if sq is None or sq in occupied:
                            for x in segment:
                                for y in segment:
                                    if x != y: segment_relation[x, y] = True
                            segment = []
                        else: segment.append(sq)
            segment_relation &= valid[:, None] & safe_destination[None, :]
            assert np.array_equal(segment_relation, a), f"segment relation mismatch {case}"
            kind = {chess.ROOK: 'rook', chess.BISHOP: 'bishop', chess.QUEEN: 'queen'}[piece]
            sw, sl = segmented_slider_predicates(occupied, safe_destination, valid, state, kind)
            assert np.array_equal(sw, w), f"segmented win predicate mismatch {case}"
            assert np.array_equal(sl, l), f"segmented loss predicate mismatch {case}"
            safe_segment_rows += int(valid.sum())
    return dict(cases=cases, legal_source_rows=checked_rows, quiet_edges=checked_edges,
                segment_source_rows=safe_segment_rows, status='pass', library='python-chess')


def clocked_dag_checks():
    """Check topological clock DP against an independently expanded game graph.

    The toy model has quiet edges in the same tier and zeroing edges to a lower
    tier. It models an automatic deadline, or a draw claim and then a later
    automatic deadline. This is not a complete FIDE model of repetition history.
    """
    rng = np.random.default_rng(831)
    results = []
    for deadline, claim in [(8, None), (100, None), (150, 100)]:
        tiers, width = 2, 8
        quiet = rng.random((tiers, width, width)) < .14
        zero = rng.random((tiers, width, width)) < .10
        zero[0] = False
        mate = np.zeros((tiers, width), bool); mate[:, 0] = True
        quiet[:, 0] = False; zero[:, 0] = False
        # Have one genuine stalemate in every tier.
        quiet[:, 1] = False; zero[:, 1] = False
        size = tiers * width * (deadline + 1)
        draw_node = size
        expanded = np.zeros((size + 1, size + 1), bool)
        seeds = np.zeros(size + 1, np.int8)
        values = np.zeros((tiers, deadline + 1, width), np.int8)
        def ix(t, h, x): return (t * (deadline + 1) + h) * width + x
        for t in range(tiers):
            for h in range(deadline, -1, -1):
                for x in range(width):
                    pos = ix(t, h, x)
                    if mate[t, x]:
                        values[t, h, x] = seeds[pos] = -1
                        continue
                    if h == deadline:
                        continue
                    options = []
                    for y in np.flatnonzero(quiet[t, x]):
                        expanded[pos, ix(t, h + 1, y)] = True
                        options.append(-int(values[t, h + 1, y]))
                    if t:
                        for y in np.flatnonzero(zero[t, x]):
                            expanded[pos, ix(t - 1, 0, y)] = True
                            options.append(-int(values[t - 1, 0, y]))
                    # Claim before an intended quiet move reaches the threshold.
                    # Or claim immediately when the clock reaches the threshold.
                    claimable = claim is not None and (h >= claim or (h == claim - 1 and quiet[t, x].any()))
                    if options and claimable:
                        expanded[pos, draw_node] = True
                        options.append(0)
                    values[t, h, x] = max(options) if options else 0
        oracle = counter_oracle(expanded, seeds)[:size].reshape(tiers, deadline + 1, width)
        assert np.array_equal(oracle, values)
        assert np.all(values[:, deadline, 0] == -1), 'mate must precede deadline draw'
        if claim is not None:
            assert not np.any(values[:, claim:, 2:] == -1), 'claim action must prevent a nonterminal loss'
        results.append(dict(deadline=deadline, claim_threshold=claim, states=size, status='pass'))
    return results


def main():
    p = argparse.ArgumentParser(); p.add_argument('--chess-path'); p.add_argument('--output', default='validation.json')
    args = p.parse_args()
    result = dict(h100_benchmarked=False, graph_tests=random_graphs(), clocked_dag_tests=clocked_dag_checks())
    if args.chess_path: result['chess_fiber_tests'] = chess_fibers(args.chess_path)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
