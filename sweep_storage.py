"""Page-size and decode-cost tradeoffs of the full native KQvKR stream, from cached data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from adaptive_storage import encode,decode,write_archive
from lookup_storage import BatchReader
from r2_blocks import LocalRangeStore


def sweep(cache_path):
    root=Path(cache_path);values=np.concatenate([np.load(root/f'KQvKR-0-{side}-1911252.npy') for side in (0,1)])
    result=[]
    for page in (4096,16384,65536,262144,1048576):
        with tempfile.TemporaryDirectory(prefix='tb-sweep-') as directory:
            start=time.perf_counter();manifest=write_archive(values,directory,page_positions=page,
                level=19,allow_lzma=False,metric='native-Syzygy-WDL-symbols')
            encode_seconds=time.perf_counter()-start
            store=LocalRangeStore(directory);reader=BatchReader(manifest,store,data_cache_bytes=0)
            # Every native slot compared, not just sampled board queries.
            assert np.array_equal(reader.probe_many(np.arange(len(values))),values)
            # Isolate CPU decode of a representative nonconstant block.
            begin=len(values)//3;block=values[begin:begin+page];frame,info=encode(block,level=19,allow_lzma=False)
            times=[]
            for _ in range(11):
                t=time.perf_counter();actual=decode(frame);times.append(time.perf_counter()-t)
            assert np.array_equal(block,actual)
            row=dict(page_positions=page,archive_object_bytes=sum(o['bytes'] for o in manifest['objects']),
                outer_manifest_bytes=(Path(directory)/'manifest.json').stat().st_size,
                index_bytes=sum(o['index_bytes'] for o in manifest['objects']),
                encode_seconds=encode_seconds,representative_frame_bytes=len(frame),
                representative_decode_median_ms=float(np.median(times)*1000),
                representative_decode_p95_ms=float(np.percentile(times,95)*1000),
                full_stream_verified=True)
            result.append(row);print(json.dumps(row),flush=True)
    return dict(status='pass',material='KQvKR',native_slots=len(values),published_syzygy_bytes=20496,
        sweep=result,live_r2_tested=False,
        caveat='Archives omit chess indexing metadata and capture-probing semantics. CPU decode is local; not cloud latency. Large pages can reduce bytes while increasing cache footprint and decode cost.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-path',default='../compression_cache');a=p.parse_args()
    result=sweep(a.cache_path);Path('storage_sweep.json').write_text(json.dumps(result,indent=2)+'\n')
