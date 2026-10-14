"""Boundary/fault tests for streamed data, wide values, large IDs and frames."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import lzma
import numpy as np
import zstandard as zstd
from adaptive_storage import (candidates,encode,decode,write_archive,write_archive_pages,
                             write_constant_archive,HEADER,ArchiveReader)
from lookup_storage import BatchReader,ByteLRU
from r2_blocks import LocalRangeStore
from storage_manifest import normalize,MAX_ID


def rejects(call):
    try:call()
    except (ValueError,IndexError,IOError,struct.error,OverflowError,lzma.LZMAError,zstd.ZstdError):return
    raise AssertionError('invalid operation accepted')


def validate():
    rng=np.random.default_rng(4416);roundtrips=0;mutations=0;selected=[]
    arrays=[np.array([-(1<<63),(1<<63)-1,0,-1,1],np.int64),
            np.full(129,(1<<62)+7,np.int64),
            np.arange(4097,dtype=np.int64)+(1<<40),np.empty(0,np.int64)]
    for _ in range(80):
        n=int(rng.integers(1,2049));kind=int(rng.integers(4))
        if kind==0:x=rng.integers(-3,4,n,dtype=np.int64)
        elif kind==1:x=rng.integers(-(1<<62),1<<62,n,dtype=np.int64)
        elif kind==2:x=(np.arange(n,dtype=np.int64)//17)+(1<<40)
        else:x=np.full(n,int(rng.integers(-100,100)),np.int64)
        arrays.append(x)
    for x in arrays:
        for frame,info in candidates(x,allow_lzma=False):
            assert np.array_equal(decode(frame),x),info;roundtrips+=1
        frame,_=encode(x,allow_lzma=False);selected.append(frame)
    # Mutate protected headers, counts and checksums. Padding is canonical too.
    for frame in selected[:20]:
        for offset in list(range(32))+[len(frame)-1]:
            changed=bytearray(frame);changed[offset]^=128
            try:decode(changed)
            except Exception:mutations+=1
            else:raise AssertionError(('undetected bit mutation',offset,len(frame)))
    for frame in selected[:10]:
        for end in (0,1,15,31,len(frame)-1):rejects(lambda f=frame,e=end:decode(f[:e]))
    # Before, the decompressor ignored data after the end of a Zstd frame.
    frame=next(f for f,i in candidates(np.arange(65),allow_lzma=False) if i['compression']=='zstd')
    fields=list(HEADER.unpack_from(frame));fields[6]+=1
    rejects(lambda:decode(HEADER.pack(*fields)+frame[32:]+b'x'))
    assert len(decode(encode([])[0]))==0
    rejects(lambda:encode([float('nan')]))
    rejects(lambda:encode(np.array([1<<63],np.uint64)))
    rejects(lambda:ByteLRU(-1))
    with tempfile.TemporaryDirectory(prefix='storage-edges-') as directory:
        page=127;values=np.concatenate([np.full(page,1<<42,np.int64),
              rng.integers(-(1<<62),1<<62,page*4+3,dtype=np.int64)])
        streamed=write_archive_pages((values[b:b+page] for b in range(0,len(values),page)),directory,
                 page_positions=page,index_page_records=2,max_pages_per_object=2,object_bytes=4096,allow_lzma=False)
        store=LocalRangeStore(directory);reader=BatchReader(streamed,store,max_bytes=64,
             data_cache_bytes=128,index_cache_bytes=64,directory_cache_bytes=64,max_batch_decoded_bytes=128,max_parallel_reads=3)
        assert np.array_equal(reader.probe_many(np.arange(len(values))),values)
        assert len(reader.probe_many([]))==0
        assert reader.stats['max_staged_decoded_bytes']<=page*8
        rejects(lambda:reader.peek(.5))
        rejects(lambda:reader.peek(True))
        # An invalid producer must not replace a previously committed root.
        old=(Path(directory)/'manifest.json').read_bytes()
        rejects(lambda:write_archive_pages(iter([np.arange(page),np.arange(5),np.arange(page)]),directory,
              page_positions=page,max_pages_per_object=1,allow_lzma=False))
        assert (Path(directory)/'manifest.json').read_bytes()==old
        for field,value in [('positions',True),('positions',len(values)+1),('page_positions',0),('index_page_records',0)]:
            bad=copy.deepcopy(streamed);bad[field]=value;rejects(lambda b=bad:BatchReader(b,store))
        bad=copy.deepcopy(streamed);bad['objects'][1]['start']+=1;rejects(lambda:normalize(bad))
        # Verify empty archives and malformed input shapes without cloud reads.
        empty=write_archive([],Path(directory)/'empty');empty_store=LocalRangeStore(Path(directory)/'empty')
        assert len(BatchReader(empty,empty_store).probe_many([]))==0;assert empty_store.requests==0
        rejects(lambda:write_archive(np.empty((0,2),np.int64),Path(directory)/'invalid'))
        constant=write_constant_archive(-(1<<63),MAX_ID,Path(directory)/'huge')
        wire=json.loads((Path(directory)/'huge'/'manifest.json').read_text())
        assert isinstance(wire['positions'],str) and isinstance(wire['constant'],str)
        constant_store=LocalRangeStore(Path(directory)/'huge');huge=BatchReader(wire,constant_store)
        ids=np.array([0,(1<<53)+123,MAX_ID-1],np.int64)
        assert np.array_equal(huge.probe_many(ids),np.full(3,-(1<<63),np.int64))
        assert huge.peek(int(ids[1]))==-(1<<63) and constant_store.requests==0
        rejects(lambda:huge.probe_many([MAX_ID]))
    return dict(status='pass',arrays=len(arrays),candidate_roundtrips=roundtrips,
        header_and_payload_mutations_rejected=mutations,int64_extremes=True,
        ids_above_2pow53_exact=True,streaming_commit_failure_preserves_old_root=True,
        fetch_cap_bytes=64,decoded_staging_budget_plus_one_page=True,
        live_r2_tested=False,h100_benchmarked=False)


if __name__=='__main__':
    result=validate();Path('storage_edge_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
