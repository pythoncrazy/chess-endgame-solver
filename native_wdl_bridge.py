"""Complete KQvKR re-encoding and batched FEN probes via python-chess.

Uses python-chess's existing canonical ranking and capture-resolution rules,
with raw symbols supplied from our archive. Private python-chess interfaces
are deliberately isolated here: this is a version-sensitive research bridge.
Other material dependencies still use supplied local Syzygy files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from adaptive_storage import write_archive
from lookup_storage import BatchReader
from r2_blocks import LocalRangeStore


class NeedSymbol(Exception):
    def __init__(self,owner,index):self.owner=owner;self.index=index


def metadata_bytes(meta):
    return (json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n').encode()


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        while True:
            block=file.read(1<<20)
            if not block:break
            digest.update(block)
    return digest.hexdigest()


def write_metadata(meta,directory):
    import os,tempfile
    directory=Path(directory);raw=metadata_bytes(meta);checksum=hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    filename='chess-index-'+checksum+'.json'
    with tempfile.NamedTemporaryFile(dir=directory,prefix='index-',delete=False) as file:
        temporary=Path(file.name)
        try:file.write(raw);file.flush();os.fsync(file.fileno())
        except BaseException:temporary.unlink(missing_ok=True);raise
    try:os.replace(temporary,directory/filename)
    finally:temporary.unlink(missing_ok=True)
    return dict(material=meta['material'],index_sha256=checksum,index_file=filename,rules='syzygy-wdl50-at-reset-v1')


def bind_metadata(meta,manifest,directory):
    from adaptive_storage import commit_manifest
    manifest['chess_domain']=write_metadata(meta,directory)
    commit_manifest(manifest,directory)
    return manifest


def load_metadata(directory,manifest):
    domain=manifest.get('chess_domain')
    if domain is None:
        if manifest['format']=='exact-blocks-v3':raise ValueError('chess companion binding required')
        # Legacy example files remain readable, with their original limitations.
        return json.loads((Path(directory)/'chess_index.json').read_text())
    import re
    if not isinstance(domain['index_sha256'],str) or not re.fullmatch('[a-f0-9]{64}',domain['index_sha256']):
        raise ValueError('invalid index checksum')
    filename='chess-index-'+domain['index_sha256']+'.json'
    if filename!=domain['index_file']:raise ValueError('index identity mismatch')
    raw=(Path(directory)/filename).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=domain['index_sha256']:raise ValueError('chess metadata checksum mismatch')
    return json.loads(raw)


def metadata(source,chess_path):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess.syzygy
    table=chess.syzygy.WdlTable(str(source));table.init_table_wdl()
    result=dict(material=Path(source).stem,pieces=table.pieces,norm=table.norm,factor=table.factor,
                stream_sizes=table.tb_size[:2],source_sha256=file_sha256(source),
                metric='Syzygy WDL5, standard capture-based probing',
                source='https://tablebase.lichess.ovh/tables/standard/'+('3-4-5-wdl' if table.num<=5 else f'{table.num}-wdl')+'/'+Path(source).name,
                has_pawns=table.has_pawns,streams=[])
    offset=0
    if table.has_pawns:
        result['pawn_files']=[]
        for f,data in enumerate(table.files):
            result['pawn_files'].append(dict(pieces=data.pieces,norm=data.norm,factor=data.factor))
            for side in sorted(data.precomp):
                count=table.tb_size[2*f+side]
                result['streams'].append(dict(file=f,side=side,count=count,offset=offset));offset+=count
    else:
        for side in sorted(table.precomp):
            count=table.tb_size[side]
            result['streams'].append(dict(file=0,side=side,count=count,offset=offset));offset+=count
    table.close();return result


def adapter(meta,reader):
    import chess.syzygy
    domain=reader.manifest.get('chess_domain')
    if domain is not None:
        if domain['material']!=meta['material'] or domain['index_sha256']!=hashlib.sha256(metadata_bytes(meta)).hexdigest():
            raise ValueError('wrong chess metadata companion')
        if domain['rules']!='syzygy-wdl50-at-reset-v1':raise ValueError('unsupported chess rules')
    positions=sum(s['count'] for s in meta['streams']) if 'streams' in meta else sum(meta['stream_sizes'])
    if reader.manifest['positions']!=positions:raise ValueError('wrong chess archive size')
    class Reencoded(chess.syzygy.WdlTable):
        def __init__(self):
            super().__init__('nonexistent/'+meta['material']+'.rtbw')
            if self.num>7:raise ValueError('python-chess bridge supports at most seven pieces')
            self.initialized=True;self.pieces={int(k):v for k,v in meta['pieces'].items()}
            self.norm=meta['norm'];self.factor=meta['factor'];self.precomp={0:0,1:1}
            self.offsets={}
            if 'streams' in meta:
                self.offsets={(s['file'],s['side']):s['offset'] for s in meta['streams']}
                if self.has_pawns:
                    self.files=[]
                    for f,entry in enumerate(meta['pawn_files']):
                        data=chess.syzygy.PawnFileData()
                        for field in ('pieces','norm','factor'):
                            setattr(data,field,{int(k):v for k,v in entry[field].items()})
                        data.precomp={s['side']:(f,s['side']) for s in meta['streams'] if s['file']==f}
                        self.files.append(data)
                else:self.precomp={s['side']:(0,s['side']) for s in meta['streams']}
            from contextvars import ContextVar
            self.symbol_context=ContextVar('tablebase-symbols-'+meta['material'],default=None)
        @property
        def symbols(self):
            value=self.symbol_context.get()
            if value is None:value={};self.symbol_context.set(value)
            return value
        @symbols.setter
        def symbols(self,value):self.symbol_context.set(value)
        def init_table_wdl(self):pass  # No original payload or file opened.
        def decompress_pairs(self,side,index):
            absolute=index+(self.offsets[side] if self.offsets else meta['stream_sizes'][0] if side else 0)
            if absolute in self.symbols:return self.symbols[absolute]
            hot=reader.peek(absolute)
            if hot is not None:return hot
            raise NeedSymbol(self,absolute)
    return Reencoded()


def export_material(material,chess_path,table_path,cache_path,archive_path,page_positions=65536):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess.syzygy
    source=Path(table_path)/(material+'.rtbw');meta=metadata(source,chess_path)
    table=chess.syzygy.WdlTable(str(source));table.init_table_wdl()
    cache=Path(cache_path);cache.mkdir(parents=True,exist_ok=True);values=[]
    for stream in meta['streams']:
        f,side,count=stream['file'],stream['side'],stream['count']
        file=cache/f"{material}-{f}-{side}-{count}-{meta['source_sha256']}.npy"
        proof=file.with_suffix('.provenance.json');reuse=False
        if file.exists() and proof.exists():
            try:
                record=json.loads(proof.read_text())
                reuse=record['source_sha256']==meta['source_sha256'] and record['npy_sha256']==file_sha256(file)
            except (ValueError,KeyError):reuse=False
        if reuse:x=np.load(file,mmap_mode='r')
        else:
            data=table.files[f].precomp[side] if table.has_pawns else table.precomp[side]
            import tempfile,os
            with tempfile.NamedTemporaryFile(dir=cache,prefix='symbols-',suffix='.npy',delete=False) as pending:
                temporary=Path(pending.name)
            try:
                x=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.int32,shape=(count,))
                for begin in range(0,count,65536):
                    size=min(65536,count-begin)
                    x[begin:begin+size]=data.min_len if not data.idxbits else np.fromiter(
                        (table.decompress_pairs(data,i) for i in range(begin,begin+size)),np.int32,count=size)
                x.flush();del x;os.replace(temporary,file)
            finally:temporary.unlink(missing_ok=True)
            x=np.load(file,mmap_mode='r')
            proof.write_text(json.dumps(dict(source_sha256=meta['source_sha256'],npy_sha256=file_sha256(file)))+'\n')
        if len(x)!=count:raise ValueError('reference cache length mismatch')
        values.append(x);print('loaded',material,f,side,count,flush=True)
    table.close();directory=Path(archive_path);directory.mkdir(parents=True,exist_ok=True)
    # Repage at stream boundaries with at most one partial page buffered.
    from adaptive_storage import write_archive_pages
    def pages():
        pending=np.empty(0,np.int64)
        for x in values:
            begin=0
            if len(pending):
                take=min(page_positions-len(pending),len(x));pending=np.concatenate([pending,x[:take]]);begin=take
                if len(pending)==page_positions:yield pending;pending=np.empty(0,np.int64)
            while begin+page_positions<=len(x):yield x[begin:begin+page_positions];begin+=page_positions
            if begin<len(x):pending=x[begin:].copy()
        if len(pending):yield pending
    domain=write_metadata(meta,directory)
    manifest=write_archive_pages(pages(),directory,page_positions=page_positions,level=19,
                                allow_lzma=False,metric=meta['metric'],catalog_fields={'chess_domain':domain})
    return meta,manifest


def probe_batch(tablebase,reencoded,reader,boards,metric='wdl',material_readers=None):
    function=tablebase.probe_wdl if metric=='wdl' else tablebase.probe_dtz if metric=='dtz' else None
    if function is None:raise ValueError('unknown metric')
    if any(not b.is_valid() or b.castling_rights or b.uci_variant!='chess' for b in boards):
        raise ValueError('valid standard non-castling positions required')
    mapping={reencoded:reader} if material_readers is None else material_readers
    pending=set(range(len(boards)));out=[None]*len(boards);rounds=0
    for encoded in mapping:encoded.symbols={}
    try:
        while pending:
            missing={}
            for i in sorted(pending):
                try:out[i]=function(boards[i])
                except NeedSymbol as need:
                    if need.owner not in mapping:raise ValueError('missing archive dependency reader')
                    missing.setdefault(need.owner,set()).add(need.index)
                else:pending.remove(i)
            if not missing:
                if pending:raise RuntimeError('probe made no progress')
                break
            for encoded,positions in missing.items():
                ids=np.asarray(sorted(positions),np.int64);values=mapping[encoded].probe_many(ids)
                encoded.symbols.update(zip(map(int,ids),map(int,values)))
            rounds+=1
    finally:
        for encoded in mapping:encoded.symbols={}
    return out,rounds


def query_archive(chess_path,table_path,archive_path,fens,metric='wdl',range_base_url=None):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess
    import chess.syzygy
    directory=Path(archive_path);manifest=json.loads((directory/'manifest.json').read_text())
    meta=load_metadata(directory,manifest)
    from r2_blocks import HttpRangeStore
    store=HttpRangeStore(range_base_url) if range_base_url else LocalRangeStore(directory)
    reader=BatchReader(manifest,store);encoded=adapter(meta,reader)
    with chess.syzygy.open_tablebase(str(table_path)) as tablebase:
        for key,table in list(tablebase.wdl.items()):
            if table.key in (encoded.key,encoded.mirrored_key):table.close();del tablebase.wdl[key]
        tablebase.wdl[encoded.key]=encoded;tablebase.wdl[encoded.mirrored_key]=encoded
        boards=[chess.Board(fen) for fen in fens]
        if any(not b.is_valid() or b.castling_rights for b in boards):raise ValueError('valid standard non-castling positions required')
        values,waves=probe_batch(tablebase,encoded,reader,boards,metric)
    return dict(metric='published-Syzygy-'+metric,values=values,archive_range_reads=store.requests,
                archive_bytes_read=store.bytes_read,dependency_waves=waves,
                note='Other materials and DTZ still need local Syzygy dependencies. WDL classes do not incorporate arbitrary prior repetition history or independently adjudicate a supplied nonzero clock.')


def experiment(chess_path,table_path,cache_path,output_path=None,page_positions=65536):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess
    import chess.syzygy
    source=Path(table_path)/'KQvKR.rtbw';meta=metadata(source,chess_path)
    temp=tempfile.TemporaryDirectory(prefix='native-wdl-') if output_path is None else None
    directory=Path(temp.name if temp else output_path);directory.mkdir(parents=True,exist_ok=True)
    meta,manifest=export_material('KQvKR',chess_path,table_path,cache_path,directory,page_positions)
    store=LocalRangeStore(directory);reader=BatchReader(manifest,store)
    encoded=adapter(meta,reader)
    oracle=chess.syzygy.open_tablebase(str(table_path));actual=chess.syzygy.open_tablebase(str(table_path))
    for key,table in list(actual.wdl.items()):
        if table.key in (encoded.key,encoded.mirrored_key):actual.wdl[key]=encoded;table.close()
    rng=np.random.default_rng(61271);boards=[]
    while len(boards)<2000:
        squares=rng.choice(64,4,replace=False);b=chess.Board(None)
        for square,piece,color in zip(squares,(chess.KING,chess.KING,chess.QUEEN,chess.ROOK),
                                      (chess.WHITE,chess.BLACK,chess.WHITE,chess.BLACK)):
            b.set_piece_at(int(square),chess.Piece(piece,color))
        b.turn=bool(rng.integers(2))
        if b.is_valid():boards.append(b)
    # Include color mirroring and fresh-clock values; no repetition-history claim.
    boards.extend([b.mirror() for b in boards[:500]])
    expected=[oracle.probe_wdl(b) for b in boards]
    got,rounds=probe_batch(actual,encoded,reader,boards)
    assert got==expected;first_requests=store.requests;first_bytes=store.bytes_read
    got_again,_=probe_batch(actual,encoded,reader,boards)
    assert got_again==expected;repeat_reads=store.requests-first_requests;assert repeat_reads==0
    dtz_boards=boards[:100]
    expected_dtz=[oracle.probe_dtz(b) for b in dtz_boards]
    got_dtz,dtz_rounds=probe_batch(actual,encoded,reader,dtz_boards,metric='dtz')
    assert got_dtz==expected_dtz
    files=[p for p in directory.iterdir() if p.is_file()]
    result=dict(status='pass',wdl_fens_checked=len(boards),includes_color_mirrors=True,
        dtz_fens_checked=len(dtz_boards),first_batch_origin_reads=first_requests,first_batch_bytes=first_bytes,
        repeated_batch_origin_reads=repeat_reads,wdl_dependency_waves=rounds,dtz_dependency_waves=dtz_rounds,
        archive_bytes_including_chess_metadata=sum(p.stat().st_size for p in files),
        published_original_wdl_bytes=source.stat().st_size,reencoded_original_file_opened=False,
        live_r2_tested=False,h100_benchmarked=False,
        caveat='WDL archive is complete for KQvKR native slots. Other material dependencies and DTZ data still come from local original Syzygy files. DTZ checks preserve published probe semantics, not a new exact-ply generator. Private python-chess APIs need version pinning.')
    actual.close();oracle.close()
    if temp:temp.cleanup()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chess-path',required=True);p.add_argument('--table-path',required=True)
    p.add_argument('--cache-path',default='../compression_cache');p.add_argument('--archive-path');p.add_argument('--page-positions',type=int,default=65536)
    p.add_argument('--fen',action='append');p.add_argument('--metric',choices=['wdl','dtz'],default='wdl')
    p.add_argument('--range-base-url',help='Optional already-deployed immutable range endpoint')
    p.add_argument('--export-material',help='Export complete native WDL streams for one reference material')
    args=p.parse_args()
    if args.export_material:
        if not args.archive_path:p.error('--export-material requires --archive-path')
        meta,manifest=export_material(args.export_material,args.chess_path,args.table_path,args.cache_path,args.archive_path,args.page_positions)
        result=dict(status='exported',material=meta['material'],positions=manifest['positions'])
    elif args.fen:
        if not args.archive_path:p.error('--fen requires --archive-path')
        result=query_archive(args.chess_path,args.table_path,args.archive_path,args.fen,args.metric,args.range_base_url)
    else:
        result=experiment(args.chess_path,args.table_path,args.cache_path,args.archive_path,args.page_positions)
        Path('native_bridge_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
