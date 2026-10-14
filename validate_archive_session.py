"""Persistent multi-material cache/session contracts, checked with published references."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def validate(chess_path,table_path,pawn_archives):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess,chess.syzygy
    from archive_session import ArchiveSession
    fens=['8/8/8/3pP3/8/6k1/8/K7 w - d6 0 1',
          '8/8/8/8/8/7k/5P2/K7 w - - 0 1',
          '8/8/8/4pP2/8/6k1/8/K7 w - e6 0 1']
    with chess.syzygy.open_tablebase(str(table_path)) as oracle:
        expected=[oracle.probe_wdl(chess.Board(fen)) for fen in fens]
    with ArchiveSession([Path(pawn_archives)/m for m in ('KPvK','KPvKP')],table_path,
                        compressed_cache_bytes=16384,page_cache_bytes=16384) as session:
        first=session.probe_fens(fens);second=session.probe_fens(fens)
        assert first['values']==expected and second['values']==expected
        assert session.compressed_cache.size<=16384
        assert session.probe_fens([])['values']==[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda _:session.probe_fens(fens)['values'],range(16)))
        assert all(value==expected for value in results)
        for invalid in ('bad FEN','r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'):
            try:session.probe_fens([invalid])
            except ValueError:pass
            else:raise AssertionError('invalid/unsupported query accepted')
        assert session.probe_fens(fens)['values']==expected
    try:session.probe_fens(fens)
    except ValueError:pass
    else:raise AssertionError('closed session accepted query')
    return dict(status='pass',concurrent_batches=16,materials=2,global_compressed_cache_bytes=16384,invalid_query_cleanup=True,closed_session_rejected=True,live_r2_tested=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chess-path',required=True);p.add_argument('--table-path',required=True);p.add_argument('--pawn-archives',required=True);args=p.parse_args()
    result=validate(args.chess_path,args.table_path,args.pawn_archives);Path('archive_session_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
