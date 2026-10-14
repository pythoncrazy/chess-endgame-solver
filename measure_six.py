"""Published six-piece metadata, sampled codec costs and native Rust probe oracle.

This does NOT generate a six-piece material. Native symbols are the codes of
the stored stream, with fillers. They are not final WDL answers for legal
positions.
"""
import argparse,hashlib,json,sys,time,tempfile,re
from pathlib import Path
import numpy as np
from adaptive_storage import encode,decode
from native_range_catalog import publish

def measure(chess_path,table_path,object_directory,samples_per_stream=16384):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess,chess.syzygy
    root=Path(__file__).parent;tables=Path(table_path);rng=np.random.default_rng(14072)
    materials=[];fens=[];expected=[];move_hist={};probe_seconds=0
    with chess.syzygy.open_tablebase(str(tables)) as oracle:
        for material in ('KQBNvKR','KNNvKNN'):
            source=tables/(material+'.rtbw');table=chess.syzygy.WdlTable(str(source));table.init_table_wdl()
            streams=[]
            for side,data in sorted(table.precomp.items()):
                n=table.tb_size[side];page=samples_per_stream//4;start=time.perf_counter();values=[];spans=[]
                for begin in np.linspace(0,n-page,4,dtype=np.int64):
                    begin=int(begin);spans.append([begin,page]);values.append(np.fromiter((table.decompress_pairs(data,i) for i in range(begin,begin+page)),np.int64,count=page))
                wall=time.perf_counter()-start;frame_sizes=[]
                for block in values:
                    frame,info=encode(block,allow_lzma=False,level=12);assert np.array_equal(decode(frame),block);frame_sizes.append(info)
                streams.append(dict(side=side,native_slots=n,sampled_slots=samples_per_stream,spans=spans,source_decode_seconds=wall,frames=frame_sizes))
            table.close();source_bytes=source.stat().st_size;total=sum(s['native_slots'] for s in streams)
            row=dict(material=material,source_url=f'https://tablebase.lichess.ovh/tables/standard/6-wdl/{source.name}',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_bytes=source_bytes,native_slots=total,
                     full_native_file_bits_per_slot=8*source_bytes/total,streams=streams,
                     micro_index_32byte_record_floor_16384_positions=32*((total+16383)//16384),
                     note='Source entire file size measured; custom frame costs are stratified samples only, not full re-encoding or final legal WDL entropy.')
            materials.append(row)
            pieces=[(chess.KING,True),(chess.KING,False)]+([(chess.QUEEN,True),(chess.BISHOP,True),(chess.KNIGHT,True),(chess.ROOK,False)] if material=='KQBNvKR' else [(chess.KNIGHT,True),(chess.KNIGHT,True),(chess.KNIGHT,False),(chess.KNIGHT,False)])
            count=0;start=time.perf_counter()
            while count<500:
                board=chess.Board(None);squares=rng.choice(64,6,replace=False)
                for sq,(kind,color) in zip(squares,pieces):board.set_piece_at(int(sq),chess.Piece(kind,color))
                board.turn=bool(rng.integers(2))
                if not board.is_valid():continue
                value=oracle.probe_wdl(board);fens.append(board.fen());expected.append(value);count+=1
                for move in board.legal_moves:
                    role=board.piece_type_at(move.from_square);label=('capture' if board.is_capture(move) else 'quiet')+'_'+str(role)
                    move_hist[label]=move_hist.get(label,0)+1
            probe_seconds+=time.perf_counter()-start
    (root/'six_probe_queries.json').write_text(json.dumps(dict(fens=fens,values=expected),separators=(',',':')))
    catalog=publish(sorted(tables.glob('*.rtbw')),object_directory)
    (root/'six_source_catalog.json').write_text(json.dumps(dict(files=[dict(name=f['name'],bytes=f['size'],source_sha256=f['key'][:-6]) for f in catalog['files']]),indent=2)+'\n')
    listing=Path('scratch_six_wdl_listing.html')
    listed=None
    if listing.exists():
        records=re.findall(r'href="([^"/]+\.rtbw)"[^\n]*?([0-9]+)\s*\n',listing.read_text())
        listed=dict(url='https://tablebase.lichess.ovh/tables/standard/6-wdl/',files=len(records),bytes=sum(int(n) for _,n in records),listing_sha256=hashlib.sha256(listing.read_bytes()).hexdigest(),retrieved_date='2026-10-13')
    result=dict(status='pass',materials=materials,legal_six_piece_fens=len(fens),independent_python_probe_seconds=probe_seconds,legal_move_histogram=move_hist,
                native_wrapper_source_bytes=sum(f['size'] for f in catalog['files']),native_wrapper_hash_bytes=sum(f['index_bytes'] for f in catalog['files']),
                native_wrapper_catalog_bytes=(Path(object_directory)/'native-catalog.json').stat().st_size,complete_six_wdl_mirror_listing=listed,
                full_six_piece_material_generated=False,h100_benchmarked=False,live_r2_tested=False)
    (root/'six_measurements.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chess-path',required=True);p.add_argument('--table-path',required=True);p.add_argument('--object-directory',default='../native_six_objects');a=p.parse_args();measure(a.chess_path,a.table_path,a.object_directory)
