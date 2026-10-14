"""Reusable FEN lookup session with shared bounded caches across materials."""
import json
from pathlib import Path
from lookup_storage import BatchReader,ByteLRU
from r2_blocks import LocalRangeStore,HttpRangeStore
from singleflight_store import SingleFlightStore
from aligned_ranges import AlignedRangeStore
from native_wdl_bridge import load_metadata,adapter,probe_batch
from aligned_ranges import RangeLRU
import threading


class ArchiveSession:
    def __init__(self,archive_directories,dependency_directory,range_base_url=None,
                 page_cache_bytes=64<<20,compressed_cache_bytes=64<<20,
                 index_cache_bytes=8<<20,directory_cache_bytes=4<<20,tile_bytes=16384):
        import chess.syzygy
        self.tablebase=chess.syzygy.open_tablebase(str(dependency_directory))
        self.mapping={};self.origins=[];self.closed=False
        self.page_cache=ByteLRU(page_cache_bytes);self.index_cache=ByteLRU(index_cache_bytes)
        self.directory_cache=ByteLRU(directory_cache_bytes)
        self.compressed_cache=RangeLRU(compressed_cache_bytes);self.compressed_lock=threading.Lock()
        try:
            for directory in map(Path,archive_directories):
                manifest=json.loads((directory/'manifest.json').read_text());meta=load_metadata(directory,manifest)
                origin=HttpRangeStore(range_base_url) if range_base_url else LocalRangeStore(directory)
                store=SingleFlightStore(origin,cache_bytes=compressed_cache_bytes)
                store.cache=self.compressed_cache;store.lock=self.compressed_lock
                if tile_bytes:store=AlignedRangeStore(store,{o['key']:o['bytes'] for o in manifest['objects']},tile_bytes)
                reader=BatchReader(manifest,store)
                reader.data=self.page_cache;reader.index=self.index_cache;reader.directory=self.directory_cache
                encoded=adapter(meta,reader)
                if any(existing.key==encoded.key for existing in self.mapping):raise ValueError('duplicate material archive')
                for key,table in list(self.tablebase.wdl.items()):
                    if table.key in (encoded.key,encoded.mirrored_key):table.close();del self.tablebase.wdl[key]
                self.tablebase.wdl[encoded.key]=encoded;self.tablebase.wdl[encoded.mirrored_key]=encoded
                self.mapping[encoded]=reader;self.origins.append(origin)
        except BaseException:self.close();raise
    def probe_fens(self,fens,metric='wdl'):
        import chess
        if self.closed:raise ValueError('session closed')
        boards=[chess.Board(fen) for fen in fens]
        if not self.mapping:raise ValueError('no archives configured')
        first=next(iter(self.mapping));before=sum(s.requests for s in self.origins)
        values,waves=probe_batch(self.tablebase,first,self.mapping[first],boards,metric,self.mapping)
        return dict(values=values,dependency_waves=waves,transport_reads=sum(s.requests for s in self.origins)-before,
                    metric='published-Syzygy-'+metric,
                    note='Archive transport reads only. Other material dependencies and DTZ data remain local; WDL assumes a reset halfmove clock.')
    def close(self):
        if not self.closed:self.tablebase.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
