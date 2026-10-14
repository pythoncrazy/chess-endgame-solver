"""Deduplicate concurrent immutable byte-range reads with a bounded cache."""
from concurrent.futures import Future
import threading
from aligned_ranges import RangeLRU


class SingleFlightStore:
    def __init__(self,store,cache_bytes=64<<20):
        self.store=store;self.cache=RangeLRU(cache_bytes);self.lock=threading.Lock();self.pending={}
        self.cache_hits=0;self.joined_reads=0;self.origin_reads=0
    def get_range(self,key,offset,length):
        token=(key,offset,length)
        with self.lock:
            cached=self.cache.get(token)
            if cached is not None:self.cache_hits+=1;return cached
            future=self.pending.get(token)
            if future is None:
                future=Future();self.pending[token]=future;owner=True;self.origin_reads+=1
            else:self.joined_reads+=1;owner=False
        if not owner:return future.result()
        try:
            raw=self.store.get_range(key,offset,length)
            if len(raw)!=length:raise IOError('short range read')
            with self.lock:self.cache.put(token,raw)
            future.set_result(raw);return raw
        except BaseException as error:
            future.set_exception(error);raise
        finally:
            with self.lock:self.pending.pop(token,None)


def validate():
    from concurrent.futures import ThreadPoolExecutor
    import time
    class SlowStore:
        def __init__(self):self.calls=0;self.fail=False
        def get_range(self,key,offset,length):
            self.calls+=1;time.sleep(.01)
            if self.fail:raise IOError('injected failure')
            return bytes((offset+i)%256 for i in range(length))
    origin=SlowStore();store=SingleFlightStore(origin,1024)
    gate=threading.Barrier(16)
    def query(_):gate.wait();return store.get_range('immutable-object',7,128)
    with ThreadPoolExecutor(max_workers=16) as pool:results=list(pool.map(query,range(16)))
    assert all(r==results[0] for r in results);assert origin.calls==1;assert not store.pending
    assert store.get_range('immutable-object',7,128)==results[0];assert origin.calls==1
    joined=store.joined_reads;origin.fail=True
    try:store.get_range('different-object',0,10)
    except IOError:pass
    else:raise AssertionError('failure swallowed')
    assert not store.pending
    origin.fail=False;assert len(store.get_range('different-object',0,10))==10
    return dict(status='pass',concurrent_callers=16,underlying_reads_for_identical_range=1,
                joined_inflight_reads=joined,failure_retry_passed=True,live_r2_tested=False)


if __name__=='__main__':
    import json
    from pathlib import Path
    result=validate();Path('singleflight_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
