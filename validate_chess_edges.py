"""Pawn archive probes, special moves and lossless game-state boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np


def validate(chess_path,table_path,pawn_archives):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess
    import chess.syzygy
    from chess_state import snapshot,restore,game_snapshot,restore_game,repetition_token,repetition_counts,can_claim_threefold,cache_identity
    from native_wdl_bridge import adapter,probe_batch,load_metadata
    from lookup_storage import BatchReader,ByteLRU
    from r2_blocks import LocalRangeStore
    rng=np.random.default_rng(6701);states=0
    # Full standard board and selected Chess960 initial arrangements.
    boards=[chess.Board()]
    for number in (0,17,518,959):
        b=chess.Board.from_chess960_pos(number);boards.append(b)
    for b in boards:
        for _ in range(80):
            raw=snapshot(b);restored=restore(raw)
            assert snapshot(restored)==raw
            assert set(restored.legal_moves)==set(b.legal_moves)
            restored,counts,known=restore_game(game_snapshot(b,True))
            assert known and counts==repetition_counts(b)
            assert can_claim_threefold(restored,counts)==b.can_claim_threefold_repetition()
            states+=1
            moves=list(b.legal_moves)
            if not moves:break
            b.push(moves[int(rng.integers(len(moves)))])
    repetition=chess.Board();cycle=['g1f3','g8f6','f3g1','f6g8']
    checked_claims=0
    for i in range(16):
        repetition.push_uci(cycle[i%4]);restored,counts,known=restore_game(game_snapshot(repetition,True))
        assert can_claim_threefold(restored,counts)==repetition.can_claim_threefold_repetition()
        assert (counts[repetition_token(restored)]>=5)==repetition.is_fivefold_repetition();checked_claims+=1
    same_fen=chess.Board(repetition.fen())
    assert cache_identity(repetition,'fide-value','optional50-auto75',True)!=cache_identity(same_fen,'fide-value','optional50-auto75',True)
    try:cache_identity(same_fen,'fide-value','optional50-auto75')
    except ValueError:pass
    else:raise AssertionError('unknown history accepted as complete')
    # Fullmove ordinal should not create needless value-cache misses.
    different_ordinal=same_fen.copy();different_ordinal.fullmove_number+=100
    assert cache_identity(same_fen,'wdl','v1',history_aware=False)==cache_identity(different_ordinal,'wdl','v1',history_aware=False)
    assert cache_identity(same_fen,'wdl','v1',history_aware=False)!=cache_identity(same_fen,'dtz','v1',history_aware=False)
    base=chess.Board('8/8/8/8/8/7k/4r3/KQ6 w - - 0 1')
    for clock in (0,99,100,149,150,151):
        base.halfmove_clock=clock;restored=restore(snapshot(base))
        assert restored.can_claim_fifty_moves()==base.can_claim_fifty_moves()
        assert restored.is_seventyfive_moves()==base.is_seventyfive_moves()
    mate=chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 150 1')
    stale=chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 150 1')
    assert mate.is_checkmate() and restore(snapshot(mate)).outcome().termination==chess.Termination.CHECKMATE
    assert stale.is_stalemate() and restore(snapshot(stale)).outcome().termination==chess.Termination.STALEMATE
    castle=chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    assert len([m for m in castle.legal_moves if castle.is_castling(m)])==2
    assert set(restore(snapshot(castle)).legal_moves)==set(castle.legal_moves)
    no_rights=castle.copy();no_rights.castling_rights=0
    assert cache_identity(castle,'wdl','v1',history_aware=False)!=cache_identity(no_rights,'wdl','v1',history_aware=False)
    pinned=chess.Board('k7/8/8/r2pP2K/8/8/8/8 w - d6 0 1')
    assert pinned.is_valid() and pinned.has_pseudo_legal_en_passant() and not pinned.has_legal_en_passant()
    no_ep=pinned.copy();no_ep.ep_square=None
    assert snapshot(pinned)!=snapshot(no_ep) and repetition_token(pinned)==repetition_token(no_ep)
    double=chess.Board('8/8/8/8/8/7k/5P2/K7 w - - 0 1');double.push_uci('f2f4')
    assert restore(snapshot(double)).ep_square==chess.F3
    oracle=chess.syzygy.open_tablebase(str(table_path));actual=chess.syzygy.open_tablebase(str(table_path));adapters=[]
    for material in ('KPvK','KPvKP'):
        directory=Path(pawn_archives)/material;manifest=json.loads((directory/'manifest.json').read_text())
        meta=load_metadata(directory,manifest);reader=BatchReader(manifest,LocalRangeStore(directory),data_cache_bytes=2*65536)
        encoded=adapter(meta,reader)
        for key,table in list(actual.wdl.items()):
            if table.key in (encoded.key,encoded.mirrored_key):table.close();actual.wdl[key]=encoded
        adapters.append((encoded,reader))
    # First compare scalar probes, then test correctly routed multi-archive
    # deferred batches, including EP dependencies into a different material.
    original_decompress={e:e.decompress_pairs for e,r in adapters}
    def direct(encoded,reader):
        def lookup(token,index):
            absolute=encoded.offsets[token]+index
            return int(reader.probe_many([absolute])[0])
        return lookup
    for encoded,reader in adapters:encoded.decompress_pairs=direct(encoded,reader)
    samples=[]
    for material in ('KPvK','KPvKP'):
        wanted=600;made=0
        while made<wanted:
            squares=rng.choice(64,3 if material=='KPvK' else 4,replace=False);b=chess.Board(None)
            for square,piece,color in zip(squares,(chess.KING,chess.KING,chess.PAWN,chess.PAWN),
                         (chess.WHITE,chess.BLACK,chess.WHITE,chess.BLACK)):
                b.set_piece_at(int(square),chess.Piece(piece,color))
            b.turn=bool(rng.integers(2))
            if b.is_valid():samples.append(b);samples.append(b.mirror());made+=1
    ep_cases=[]
    for f in (0,1,3,6,7):
        other=f+1 if f<7 else f-1
        for _ in range(50):
            b=chess.Board(None);wk,bk=map(int,rng.choice(64,2,replace=False))
            b.set_piece_at(wk,chess.Piece(chess.KING,chess.WHITE));b.set_piece_at(bk,chess.Piece(chess.KING,chess.BLACK))
            if wk in (32+f,32+other) or bk in (32+f,32+other):continue
            b.set_piece_at(32+f,chess.Piece(chess.PAWN,chess.WHITE));b.set_piece_at(32+other,chess.Piece(chess.PAWN,chess.BLACK))
            b.turn=chess.WHITE;b.ep_square=40+other
            if b.is_valid() and b.has_legal_en_passant():ep_cases.extend([b,b.mirror()])
    samples.extend(ep_cases)
    promotion_parent=chess.Board('7k/5P2/6K1/8/8/8/8/8 w - - 0 1')
    capture_parent=chess.Board('r6k/1P6/6K1/8/8/8/8/8 w - - 0 1')
    promoted=[];promotion_types=set();capture_types=set()
    for parent,target in ((promotion_parent,promotion_types),(capture_parent,capture_types)):
        samples.append(parent)
        for move in list(parent.legal_moves):
            if move.promotion:
                target.add(move.promotion);child=parent.copy();child.push(move);promoted.append(child);samples.append(child)
                assert restore(snapshot(child)).promoted==child.promoted
    assert promotion_types==capture_types=={chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT}
    matches=0;ep_changes=0
    for b in samples:
        expected=oracle.probe_wdl(b);value=actual.probe_wdl(b)
        assert value==expected,b.fen();matches+=1
    for b in ep_cases:
        no_ep=b.copy();no_ep.ep_square=None
        ep_changes+=oracle.probe_wdl(b)!=oracle.probe_wdl(no_ep)
    assert ep_changes>0, 'EP test must include value-changing cases'
    # Verify deferred batch reconstruction of symmetric pawn indexing itself.
    for encoded,reader in adapters:encoded.decompress_pairs=original_decompress[encoded]
    mapping=dict(adapters);encoded,reader=adapters[1];batch=samples
    got,waves=probe_batch(actual,encoded,reader,batch,material_readers=mapping)
    assert got==[oracle.probe_wdl(b) for b in batch]
    from concurrent.futures import ThreadPoolExecutor
    expected_batch=[oracle.probe_wdl(b) for b in batch[:32]]
    def threaded(_):
        copies=[b.copy(stack=False) for b in batch[:32]]
        values,_=probe_batch(actual,encoded,reader,copies,material_readers=mapping)
        assert values==expected_batch
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(threaded,range(16)))
    # An omitted material reader cannot be replaced with a different archive.
    for e,r in adapters:r.data=ByteLRU(0);e.symbols={}
    try:probe_batch(actual,encoded,reader,[ep_cases[0]],material_readers={encoded:reader})
    except ValueError:pass
    else:raise AssertionError('missing cross-material dependency accepted')
    assert not encoded.symbols
    for e,r in adapters:
        directory=Path(pawn_archives)/e.key
        if not directory.exists():continue
        manifest=json.loads((directory/'manifest.json').read_text())
        if 'chess_domain' not in manifest:continue
        path=directory/manifest['chess_domain']['index_file'];original=path.read_bytes()
        damaged=bytearray(original);damaged[-2]^=1;path.write_bytes(damaged)
        try:load_metadata(directory,manifest)
        except ValueError:pass
        else:raise AssertionError('corrupt chess companion accepted')
        finally:path.write_bytes(original)
    oracle.close();actual.close()
    return dict(status='pass',snapshot_move_states=states,repetition_claim_states=checked_claims,
        wdl_positions_checked=matches,ep_positions=len(ep_cases),value_changing_ep_positions=ep_changes,
        promotion_types=sorted(promotion_types),capture_promotion_types=sorted(capture_types),
        pawn_batch_positions=len(batch),pawn_dependency_waves=waves,concurrent_batches_checked=16,
        cross_material_dependency_routing=True,missing_dependency_rejected=True,
        chess_companion_corruption_rejected=True,castling_snapshot_and_moves_preserved=True,
        castling_tables_solved=False,checkmate_precedes_75_move_draw=True,history_sensitive_keys=True,
        full_fide_history_tablebase_solved=False,live_r2_tested=False,h100_benchmarked=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chess-path',required=True);p.add_argument('--table-path',required=True)
    p.add_argument('--pawn-archives',required=True);a=p.parse_args()
    result=validate(a.chess_path,a.table_path,a.pawn_archives);Path('chess_edge_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
