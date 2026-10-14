"""Validate the exact quotient solver on cyclic games and complete KQvK boards.

KQvK uses standard chess movement and all legal canonical placements. It has
both sides to move, king moves, queen moves, checkmate, stalemate and draws by
queen capture. Castling and en passant are not possible. The independent
Syzygy WDL comparison is optional. Terminal DTM ignores the halfmove clock and
earlier history.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from exact_quotient import sparse_oracle, solve_quotient


def random_checks():
    rng = np.random.default_rng(9901)
    result = []
    for case in range(32):
        n = 31 + case
        a = rng.random((n,n)) < .09
        terminal = rng.choice(n, 5, replace=False); a[terminal] = False
        seeds = np.zeros(n,np.int8); seeds[terminal] = rng.choice([-1,0,1],len(terminal))
        graph = [np.flatnonzero(row).tolist() for row in a]
        expected_w, expected_d = sparse_oracle(graph, seeds)
        w,d,stats = solve_quotient(graph,seeds)
        assert np.array_equal(w,expected_w) and np.array_equal(d,expected_d)
        result.append(stats)
    # Lift a small quotient into many concrete nodes, with cycles. These nodes
    # have different concrete successors but the same sets of successor classes.
    base = [[1,2],[3],[2],[4],[]]; seed = [0,0,0,0,-1]
    copies = 128
    graph=[]; seeds=[]
    for block, children in enumerate(base):
        for copy in range(copies):
            graph.append(sorted({j*copies+(copy*17+j)%copies for j in children}))
            seeds.append(seed[block])
    ew,ed = sparse_oracle(graph,seeds); w,d,stats=solve_quotient(graph,seeds)
    assert np.array_equal(w,ew) and np.array_equal(d,ed)
    assert stats['quotient_nodes']==5
    return dict(random_cases=len(result), lifted_example=stats,
                random_node_reductions=[x['node_reduction'] for x in result])


def chess_check(chess_path, table_path=None, on_solved=None):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess
    maps=[]
    for t in range(8):
        row=[]
        for sq in range(64):
            f,r=sq%8,sq//8
            if t&1: f,r=r,f
            if t&2: f=7-f
            if t&4: r=7-r
            row.append(r*8+f)
        maps.append(row)
    def canon(wk,bk,q):
        return min((m[wk],m[bk],m[q]) for m in maps)
    start=time.perf_counter()
    king_pairs=[]
    for wk in range(64):
        for bk in range(64):
            if chess.square_distance(wk,bk)<=1: continue
            if (wk,bk)==min((m[wk],m[bk]) for m in maps): king_pairs.append((wk,bk))
    assert len(king_pairs)==462
    nodes=[]; lookup={}; board=chess.Board(None)
    for wk,bk in king_pairs:
        board.set_piece_at(wk,chess.Piece(chess.KING,chess.WHITE))
        board.set_piece_at(bk,chess.Piece(chess.KING,chess.BLACK))
        for q in range(64):
            if q in (wk,bk) or canon(wk,bk,q)!=(wk,bk,q): continue
            board.set_piece_at(q,chess.Piece(chess.QUEEN,chess.WHITE))
            for stm in (chess.WHITE,chess.BLACK):
                board.turn=stm
                if board.is_valid():
                    node=(wk,bk,q,stm);lookup[node]=len(nodes);nodes.append(node)
            board.remove_piece_at(q)
        board.clear_board()
    print('canonical KQvK nodes',len(nodes),flush=True)
    draw=len(nodes); graph=[]; seeds=[]
    tb=None
    if table_path:
        import chess.syzygy
        tb=chess.syzygy.open_tablebase(table_path)
    expected=[]; predictor=[]; observed_turn=[]
    for i,(wk,bk,q,stm) in enumerate(nodes):
        board.clear_board();board.turn=stm
        board.set_piece_at(wk,chess.Piece(chess.KING,chess.WHITE))
        board.set_piece_at(bk,chess.Piece(chess.KING,chess.BLACK))
        board.set_piece_at(q,chess.Piece(chess.QUEEN,chess.WHITE))
        children=[];capture_draw=False
        for move in list(board.legal_moves):
            board.push(move)
            qs=board.pieces(chess.QUEEN,chess.WHITE)
            if not qs:
                children.append(draw);capture_draw=True
            else:
                key=(*canon(board.king(chess.WHITE),board.king(chess.BLACK),next(iter(qs))),board.turn)
                children.append(lookup[key])
            board.pop()
        graph.append(sorted(set(children)))
        seeds.append(-1 if not children and board.is_check() else 0)
        predictor.append(0 if capture_draw or (not children and not board.is_check()) else 1 if stm==chess.WHITE else -1)
        observed_turn.append(stm)
        if tb is not None:
            value=tb.probe_wdl(board);expected.append(1 if value>0 else -1 if value<0 else 0)
        if (i+1)%10000==0:print('built',i+1,'positions',flush=True)
    graph.append([]);seeds.append(0)
    build_seconds=time.perf_counter()-start
    start=time.perf_counter(); ew,ed=sparse_oracle(graph,seeds); oracle_seconds=time.perf_counter()-start
    if expected: assert np.array_equal(ew[:-1],expected),'Syzygy WDL mismatch'
    if on_solved is not None:on_solved(nodes,graph,ew,ed)
    # Exact for this KQvK domain, which the check covers fully. This is not a
    # general theorem about queen endings with other material or any halfmove clock.
    predictor_mismatches=int(np.count_nonzero(ew[:-1]!=predictor))
    assert predictor_mismatches==0,'KQvK geometric WDL predictor mismatch'
    import zstandard as zstd
    from r2_blocks import pack
    codes=np.where(ew[:-1]==1,1,np.where(ew[:-1]==-1,2,0)).astype(np.uint8)
    raw=pack(codes); compressed=zstd.ZstdCompressor(level=19).compress(raw)
    compression=dict(raw_two_bit_bytes=len(raw),zstd_level19_bytes=len(compressed),
                     downloaded_syzygy_bytes=Path(table_path,'KQvK.rtbw').stat().st_size if table_path else None,
                     caveat='Canonical index/metadata differs. Sizes are not a like-for-like format benchmark.')
    if tb is not None:tb.close()
    start=time.perf_counter()
    w,d,stats=solve_quotient(graph,seeds,lambda r,n:print('refinement',r,'blocks',n,flush=True))
    quotient_seconds=time.perf_counter()-start
    assert np.array_equal(w,ew),'quotient WDL mismatch'
    assert np.array_equal(d,ed),'quotient DTM mismatch'
    return dict(**stats,build_seconds=build_seconds,oracle_seconds=oracle_seconds,
                quotient_seconds=quotient_seconds,max_dtm_plies=int(ed.max()),
                syzygy_positions_checked=len(expected),status='pass',
                geometric_wdl_predictor_mismatches=predictor_mismatches,compression=compression,
                caveat='Explicit graph construction costs are included; no eight-piece extrapolation justified.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--chess-path');p.add_argument('--table-path');p.add_argument('--output',default='quotient_validation.json')
    args=p.parse_args();data=dict(h100_benchmarked=False,random=random_checks())
    if args.chess_path:data['KQvK']=chess_check(args.chess_path,args.table_path)
    Path(args.output).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))


if __name__=='__main__':main()
