"""Actual five-class WDL values from sampled KNNvKP positions, losslessly."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from adaptive_storage import encode,decode


def validate(chess_path,table_path):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess
    import chess.syzygy
    rng=np.random.default_rng(7503);hist=Counter();examples={};values=[]
    with chess.syzygy.open_tablebase(str(table_path)) as table:
        while len(values)<5000:
            squares=rng.choice(64,5,replace=False);board=chess.Board(None)
            for square,piece,color in zip(squares,(chess.KING,chess.KING,chess.KNIGHT,chess.KNIGHT,chess.PAWN),
                                    (chess.WHITE,chess.BLACK,chess.WHITE,chess.WHITE,chess.BLACK)):
                board.set_piece_at(int(square),chess.Piece(piece,color))
            board.turn=bool(rng.integers(2))
            if not board.is_valid():continue
            value=table.probe_wdl(board);values.append(value);hist[value]+=1;examples.setdefault(value,board.fen())
    x=np.asarray(values,np.int64);frame,info=encode(x,allow_lzma=False)
    assert np.array_equal(decode(frame),x)
    assert set(hist)=={-2,-1,0,1,2}, 'Need actual representatives of all five classes'
    return dict(status='pass',material='KNNvKP',sampled_legal_positions=len(values),classes=dict(hist),
                examples=examples,all_five_classes_preserved=True,frame_bytes=len(frame),encoding=info,
                full_material_generated=False,h100_benchmarked=False,live_r2_tested=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chess-path',required=True);p.add_argument('--table-path',required=True);a=p.parse_args()
    result=validate(a.chess_path,a.table_path);Path('cursed_class_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
