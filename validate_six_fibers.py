"""Exact shared knight matrix and occluded slider fills vs six-piece legal moves."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np

MASK=(1<<64)-1
def shift(x,df,dr,steps):
    keep=sum(1<<sq for sq in range(64) if 0<=sq%8+df*steps<8 and 0<=sq//8+dr*steps<8)
    amount=(8*dr+df)*steps
    return ((x&keep)<<amount)&MASK if amount>=0 else (x&keep)>>-amount
def knight(x):
    result=0
    for offset,mask in [(17,0x7f7f7f7f7f7f7f7f),(15,0xfefefefefefefefe),
                        (10,0x3f3f3f3f3f3f3f3f),(6,0xfcfcfcfcfcfcfcfc),
                        (-15,0x7f7f7f7f7f7f7f7f),(-17,0xfefefefefefefefe),
                        (-6,0x3f3f3f3f3f3f3f3f),(-10,0xfcfcfcfcfcfcfcfc)]:
        result|=((x&mask)<<offset)&MASK if offset>0 else (x&mask)>>-offset
    return result
def sliders(targets,empty,piece):
    directions=[]
    if piece in (4,5):directions +=[(1,0),(-1,0),(0,1),(0,-1)]
    if piece in (3,5):directions +=[(1,1),(-1,1),(1,-1),(-1,-1)]
    result=0
    for df,dr in directions:
        g=targets;p=empty
        g|=p&shift(g,df,dr,1);p&=shift(p,df,dr,1)
        g|=p&shift(g,df,dr,2);p&=shift(p,df,dr,2)
        g|=p&shift(g,df,dr,4)
        result|=shift(g,df,dr,1)&empty
    return result
def word(bits):return sum(int(x)<<i for i,x in enumerate(bits))
def decode_word(x):return np.array([(x>>i)&1 for i in range(64)],bool)

def validate(chess_path,cases=320):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess
    rng=np.random.default_rng(13067);matrix=np.zeros((64,64),np.int32)
    for source in range(64):matrix[source]=decode_word(int(chess.BB_KNIGHT_ATTACKS[source]))
    assert max(matrix.sum(1))==8
    assert max(matrix.reshape(64,16,4).sum(2).flat)<=2
    edge_count=valid_rows=checked=0;fixtures=[];piece_counts={};start=time.perf_counter()
    for case in range(cases):
        moving=(chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN)[case%4]
        color=bool((case//4)%2)
        while True:
            squares=rng.choice(64,5,replace=False).tolist()
            if chess.square_distance(squares[0],squares[1])>1:break
        # Exactly six pieces after inserting the moving piece; mixed blockers.
        board=chess.Board(None)
        fixed=[(squares[0],chess.KING,chess.WHITE),(squares[1],chess.KING,chess.BLACK),
               (squares[2],chess.ROOK,not color),(squares[3],chess.BISHOP,color),(squares[4],chess.KNIGHT,not color)]
        for sq,kind,c in fixed:board.set_piece_at(sq,chess.Piece(kind,c))
        board.turn=color;occupied=int(board.occupied)
        valid=np.zeros(64,bool);safe=np.zeros(64,bool);actual=np.zeros((64,64),bool)
        for source in range(64):
            if occupied>>source&1:continue
            board.set_piece_at(source,chess.Piece(moving,color))
            if board.is_valid():
                valid[source]=True
                for move in board.legal_moves:
                    if move.from_square==source and not board.is_capture(move):actual[source,move.to_square]=True
            safe[source]=not board.is_check();board.remove_piece_at(source)
        state=rng.integers(-1,2,64,dtype=np.int8);w=word(state==1);l=word(state==-1);d=word(safe);v=word(valid)
        reach=lambda target:knight(target) if moving==chess.KNIGHT else sliders(target,MASK^occupied,moving)
        expected=dict(winning=word((actual&(state==-1)[None,:]).any(1)),
                      nonwinning=word((actual&(state!=1)[None,:]).any(1)),legal=word(actual.any(1)))
        computed=dict(winning=reach(l&d)&v,nonwinning=reach((MASK^w)&d)&v,legal=reach(d)&v)
        assert computed==expected,(case,moving,computed,expected)
        # Independently reconstruct the full legal adjacency from target singletons.
        reconstructed=np.stack([decode_word(reach(1<<dest)&v) if safe[dest] else np.zeros(64,bool) for dest in range(64)],1)
        assert np.array_equal(actual,reconstructed)
        if moving==chess.KNIGHT:
            assert np.array_equal(actual,valid[:,None]&matrix.astype(bool)&safe[None,:])
            encoded=(safe*(1+9*(state!=1)+81*(state==-1))).astype(np.int8)
            z=matrix@encoded.astype(np.int32)
            assert z.max(initial=0)<=728
            decoded=dict(winning=word(valid&(z//81>0)),nonwinning=word(valid&((z//9)%9>0)),legal=word(valid&(z%9>0)))
            assert decoded==expected
            fp=matrix.astype(np.float16).astype(np.float32)@encoded.astype(np.float16).astype(np.float32)
            assert np.array_equal(fp,z)
            encoded16=(safe*(16*(state!=1)+(state==-1))).astype(np.int8)
            z16=matrix@encoded16.astype(np.int32)
            assert z16.max(initial=0)<=136
            assert word(valid&((z16&15)>0))==expected['winning']
            assert word(valid&(z16!=0))==expected['nonwinning']
            assert np.array_equal(matrix.astype(np.float16).astype(np.float32)@encoded16.astype(np.float16).astype(np.float32),z16)
            encoded8=(safe*(8*(state!=1)+(state==-1))).astype(np.int8)
            z8=matrix@encoded8.astype(np.int32)
            assert z8.max(initial=0)<=72
            assert word(valid&(((z8&7)>0)|(z8==72)))==expected['winning']
            assert word(valid&(z8!=0))==expected['nonwinning']
            fp8=lambda raw: 0.0 if raw==0 else (1+(raw&7)/8)*2**((raw>>3)-7)
            assert [fp8(raw) for raw in (0,0x38,0x50,0x51)]==[0,1,8,9]
            assert np.array_equal(matrix.astype(np.float32)@encoded8.astype(np.float32),z8)
        fixtures.append(dict(piece=moving,w=str(w),l=str(l),destination_safe=str(d),source_valid=str(v),occupied=str(occupied),expected={k:str(x) for k,x in expected.items()}))
        valid_rows+=int(valid.sum());edge_count+=int(actual.sum());checked+=4096;piece_counts[moving]=piece_counts.get(moving,0)+1
    out=Path(__file__).parent
    (out/'rust'/'fixtures'/'six_fibers.json').write_text(json.dumps(fixtures,separators=(',',':')))
    result=dict(status='pass',six_piece_fibers=cases,full_adjacency_cells_checked=checked,legal_source_rows=valid_rows,quiet_edges=edge_count,piece_cases=piece_counts,
                shared_knight_geometry=True,independent_columns_not_padding=True,knight_max_degree=8,knight_base9_int8_rhs=[1,10,91],max_int32_sum=728,
                knight_exact_2of4_eligible=True,sparse_gpu_api_implemented=False,division_free_base16_rhs=[0,16,17],division_free_max_int32_sum=136,static_legal_mask_required=True,occluded_slider_fill=True,integer_and_fp16_model_match=True,cpu_seconds=time.perf_counter()-start,
                fp8_exact_rhs=[0,8,9],fp8_max_sum=72,fp8_eight_loss_carry_detected=True,fp8_device_arithmetic_tested=False,
                full_six_piece_material_solved=False,h100_benchmarked=False)
    (out/'six_fiber_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chess-path',required=True);p.add_argument('--cases',type=int,default=320);a=p.parse_args();validate(a.chess_path,a.cases)
