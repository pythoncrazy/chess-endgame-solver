"""Verify lookup reductions, index pages, cache churn and exact results."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from adaptive_storage import write_archive,ArchiveReader
from lookup_storage import BatchReader,coalesce
from r2_blocks import LocalRangeStore


def validate():
    rng=np.random.default_rng(5113);page=1024
    values=rng.integers(-2,3,64*page+31,dtype=np.int32)
    values[10*page:12*page]=0
    with tempfile.TemporaryDirectory(prefix='tb-lookup-') as directory:
        manifest=write_archive(values,directory,page_positions=page,index_page_records=8,
                               object_bytes=1<<20,level=6,allow_lzma=False,metric='WDL5')
        assert len(manifest['objects'])==1
        ids=np.arange(8*page,dtype=np.int64)
        old_store=LocalRangeStore(directory);old=ArchiveReader(manifest,old_store)
        assert np.array_equal(old.probe_many(ids),values[ids])
        new_store=LocalRangeStore(directory);new=BatchReader(manifest,new_store,max_parallel_reads=4)
        t=time.perf_counter();assert np.array_equal(new.probe_many(ids),values[ids]);cpu=time.perf_counter()-t
        cold_reads=new_store.requests;cold_bytes=new_store.bytes_read
        assert cold_reads==3  # Directory, one index page, one coalesced payload.
        assert old_store.requests==9
        before=new_store.requests
        assert np.array_equal(new.probe_many(ids),values[ids]);assert new_store.requests==before
        # Warm metadata but uncached data costs one GET for a single new page.
        new.data.items.clear();new.data.size=0;before=new_store.requests
        assert np.array_equal(new.probe_many([37]),values[[37]])
        assert new_store.requests-before==1
        compact_cache_bytes=new.data.size
        assert compact_cache_bytes<=page, 'Packed WDL cache should use at most one byte/cell here'
        # Constant pages do not fetch a frame. Warm both relevant index pages.
        new.records(0,[10,11]);before=new_store.requests
        constant_ids=np.arange(10*page,12*page)
        assert np.array_equal(new.probe_many(constant_ids),values[constant_ids])
        assert new_store.requests==before
        # Full round trip with tiny caches, many index pages, a final partial
        # page and more IDs than pages. Test serial + parallel fetch planners.
        churn=BatchReader(manifest,LocalRangeStore(directory),data_cache_bytes=page,
                          index_cache_bytes=128,directory_cache_bytes=32,max_bytes=4096,max_parallel_reads=4)
        assert np.array_equal(churn.probe_many(np.arange(len(values))),values)
        random_ids=rng.integers(0,len(values),10000)
        assert np.array_equal(churn.probe_many(random_ids),values[random_ids])
        for bad in ([-1],[len(values)],[.5]):
            try:new.probe_many(bad)
            except IndexError:pass
            else:raise AssertionError('invalid ID accepted')
        # Authenticated index-page failure, independent of the directory checksum.
        o=manifest['objects'][0];path=Path(directory)/o['key'];original=path.read_bytes()
        from adaptive_storage import OBJECT_HEADER
        damaged=bytearray(original);damaged[OBJECT_HEADER.size+o['directory_bytes']]^=1;path.write_bytes(damaged)
        try:BatchReader(manifest,LocalRangeStore(directory)).probe_many([0])
        except ValueError:pass
        else:raise AssertionError('corrupt index page accepted')
        path.write_bytes(original)
        result=dict(status='pass',positions=len(values),random_queries=len(random_ids),
             adjacent_pages=8,ordinary_cold_requests=old_store.requests,optimized_cold_requests=cold_reads,
             optimized_cold_bytes=cold_bytes,cached_batch_requests=0,warm_index_single_page_requests=1,
             warm_index_constant_page_requests=0,wdl_cache_bytes_for_one_page=compact_cache_bytes,
             local_first_batch_seconds=cpu,live_r2_tested=False,
             warning='Request counts measured with local Range store. Local CPU time is not cloud latency or whole chess-probe latency.')
    # Refuse large read amplification even where gaps are below threshold.
    separate=coalesce([(0,10,'a'),(1000,10,'b')],max_gap=4096,max_amplification=2)
    assert len(separate)==2
    contiguous=coalesce([(0,10,'a'),(10,10,'b')]);assert len(contiguous)==1
    return result


if __name__=='__main__':
    result=validate();Path('lookup_storage_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
