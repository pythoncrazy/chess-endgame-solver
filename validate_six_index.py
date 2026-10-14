"""Independent padded index/quiet move-closure fixtures, using python-chess."""
import argparse
import json
import random
import sys
from pathlib import Path

def transform(square, symmetry):
    file, rank = square % 8, square // 8
    if symmetry & 4: file, rank = rank, file
    if symmetry & 1: file = 7-file
    if symmetry & 2: rank = 7-rank
    return rank*8+file

def validate(chess_path):
    sys.path.insert(0, str(Path(chess_path).resolve()))
    import chess
    root = Path(__file__).parent
    pairs = sorted({min((transform(w,s),transform(b,s)) for s in range(8))
                    for w in range(64) for b in range(64) if chess.square_distance(w,b)>1})
    assert len(pairs)==462
    pair_ids = {pair:i for i,pair in enumerate(pairs)}
    def rank(squares, black):
        pair,s = min(((transform(squares[0],s),transform(squares[1],s)),s) for s in range(8))
        value = int(black)*462+pair_ids[pair]
        for square in squares[2:]: value=value*64+transform(square,s)
        return value,s
    maps=[]
    for w in range(64):
        for b in range(64):
            if chess.square_distance(w,b)<=1: maps.append(None)
            else:
                pair,s=min(((transform(w,s),transform(b,s)),s) for s in range(8))
                maps.append([pair_ids[pair],s])
    fixtures=[];quiet=king=0
    fens=json.loads((root/'six_probe_queries.json').read_text())['fens'][:256]
    for fen in fens:
        board=chess.Board(fen)
        pieces=board.piece_map()
        squares=[board.king(chess.WHITE),board.king(chess.BLACK)]+sorted(s for s,p in pieces.items() if p.piece_type!=chess.KING)
        value,s=rank(squares,board.turn==chess.BLACK)
        canonical=[transform(square,s) for square in squares]
        assert rank(canonical,board.turn==chess.BLACK)[0]==value
        # Independent board transformation preserves actual legal quiet moves.
        transformed=board.transform(lambda bb:sum(1<<transform(sq,s) for sq in chess.scan_forward(bb)))
        children=[]
        for move in board.legal_moves:
            if board.is_capture(move): continue
            assert move.promotion is None and not board.is_castling(move)
            axis=squares.index(move.from_square)
            target=transform(move.to_square,s)
            assert chess.Move(canonical[axis],target) in transformed.legal_moves
            child=squares.copy();child[axis]=move.to_square
            child_value,_=rank(child,board.turn!=chess.BLACK)
            canonical_child=canonical.copy();canonical_child[axis]=target
            # Residual king stabilizers allow different orientations to represent
            # the same board. Compare the chosen working child from canonical
            # coordinates, not an arbitrary original orientation's child ID.
            child_value,_=rank(canonical_child,board.turn!=chess.BLACK)
            children.append([axis,target,str(child_value)])
            quiet+=1;king+=axis<2
        fixtures.append(dict(squares=squares,black=board.turn==chess.BLACK,id=str(value),canonical=canonical,children=children))
    rng=random.Random(13068)
    ids=[0,1,462*64**4-1,462*64**4,2*462*64**4-1]+[rng.randrange(2*462*64**4) for _ in range(2048)]
    (root/'rust'/'fixtures'/'six_index.json').write_text(json.dumps(dict(maps=maps,positions=fixtures,roundtrip_ids=[str(i) for i in ids]),separators=(',',':')))
    result=dict(status='pass',canonical_king_pairs=len(pairs),king_pair_inputs=4096,padded_slots=2*462*64**4,
                legal_six_piece_positions=len(fixtures),quiet_transitions=quiet,king_transitions=king,
                random_and_boundary_roundtrip_ids=len(ids),residual_stabilizer_duplicates_retained=True,
                captures_promotions_and_castling_indexed=False,h100_benchmarked=False)
    (root/'six_index_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chess-path',required=True);args=parser.parse_args();validate(args.chess_path)
