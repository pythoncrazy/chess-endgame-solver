"""Round trips, exact residuals, integrity failures and range archive checks."""
import json
from pathlib import Path
import tempfile
import numpy as np
from adaptive_storage import candidates,encode,decode,write_archive,ArchiveReader
from r2_blocks import LocalRangeStore


def validate():
    rng=np.random.default_rng(11547);count=0;mode_counts={}
    arrays=[np.empty(0,np.int32),np.array([-2147483648,2147483647],np.int64)]
    for n in (1,2,3,7,8,9,63,64,65,127,1024,4097):
        arrays.extend([np.full(n,-2,np.int32),rng.integers(-2,4,n,dtype=np.int32),
            rng.integers(-(1<<31),1<<31,n,dtype=np.int64),
            ((np.arange(n)//64)%13).astype(np.int32),
            rng.integers(-1000,1001,n,dtype=np.int32)])
    for x in arrays:
        prediction=np.zeros(len(x),np.int32)
        for frame,info in candidates(x,prediction):
            assert np.array_equal(decode(frame,prediction),x),info
            mode_counts[info['mode']]=mode_counts.get(info['mode'],0)+1;count+=1
        frame,info=encode(x,prediction)
        assert np.array_equal(decode(frame,prediction),x)
    # Exact predictor plus corrections, with an adversarial changed predictor.
    prediction=(np.arange(65536)%101).astype(np.int32);x=prediction.copy();x[[9,255,4096,65535]]+=[1,-2,13,-77]
    frames=candidates(x,prediction)
    sparse=[f for f,info in frames if info['mode']=='predictor-sparse']
    assert sparse
    for frame in sparse:
        assert np.array_equal(decode(frame,prediction),x)
        changed=prediction.copy();changed[0]+=1
        try:decode(frame,changed)
        except ValueError:pass
        else:raise AssertionError('changed predictor accepted')
    for bad in ([1.2],[-(1<<63)-1],[1<<63]):
        try:encode(bad)
        except ValueError:pass
        else:raise AssertionError('out-of-domain value accepted')
    page=256
    values=np.concatenate([np.full(page*3,-2,np.int32),rng.integers(-200,201,page*17+31,dtype=np.int32)])
    with tempfile.TemporaryDirectory(prefix='adaptive-tb-') as directory:
        manifest=write_archive(values,directory,page_positions=page,object_bytes=2048,allow_lzma=False)
        store=LocalRangeStore(directory);reader=ArchiveReader(manifest,store,cache_bytes=2048)
        ids=rng.integers(0,len(values),10000)
        assert np.array_equal(reader.probe_many(ids),values[ids])
        assert np.array_equal(reader.probe_many(np.arange(len(values))),values)
        for bad in ([-1],[len(values)],[.5]):
            try:reader.probe_many(bad)
            except IndexError:pass
            else:raise AssertionError('bad ID accepted')
        # First three constant pages share only the one object-index range.
        const_store=LocalRangeStore(directory);const_reader=ArchiveReader(manifest,const_store)
        assert np.array_equal(const_reader.probe_many(np.arange(page*3)),values[:page*3])
        assert const_store.requests==1
        # Object-index integrity is checked before trusting offsets/constants.
        first=manifest['objects'][0];path=Path(directory)/first['key'];original=path.read_bytes()
        damaged=bytearray(original);damaged[0]^=1;path.write_bytes(damaged)
        try:ArchiveReader(manifest,LocalRangeStore(directory)).probe_many([0])
        except ValueError:pass
        else:raise AssertionError('corrupt index accepted')
        path.write_bytes(original)
        # Frame corruption detection using a new reader/cache.
        target=next((j,p,record) for j in range(len(manifest['objects']))
                    for p,record in enumerate(reader.object_index(j)) if record[1])
        j,p,entry=target;offset,length=entry[:2];obj=manifest['objects'][j];path=Path(directory)/obj['key']
        damaged=bytearray(path.read_bytes());damaged[offset+length-1]^=1;path.write_bytes(damaged)
        try:ArchiveReader(manifest,LocalRangeStore(directory)).probe_many([obj['start']+p*page])
        except Exception:pass
        else:raise AssertionError('corrupt frame accepted')
        archive=dict(positions=len(values),queries=len(ids),objects=len(manifest['objects']),
                     index_bytes=sum(o['index_bytes'] for o in manifest['objects']),constant_page_reads=const_store.requests)
    return dict(status='pass',candidate_roundtrips=count,modes=mode_counts,archive=archive,
                precise_integer_values=True,predictor_mismatch_rejected=True,corruption_detected=True,
                live_r2_tested=False,h100_benchmarked=False)


if __name__=='__main__':
    result=validate();Path('adaptive_storage_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
