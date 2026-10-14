"""Immutable two-bit WDL blocks, Zstandard frames, range reads and batch cache.

Local reference implementation. S3RangeStore supports an already-configured
S3/R2 client for reads only; this script does not create cloud resources or
upload data. This is a research format, not a Syzygy-compatible DTZ format.
"""
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import zstandard as zstd


def pack(codes):
    codes = np.asarray(codes)
    if codes.ndim != 1 or not np.isin(codes, [0,1,2,3]).all():
        raise ValueError('codes must be a one-dimensional array in [0,3]')
    padded = np.zeros((len(codes)+3)//4*4,np.uint8)
    padded[:len(codes)] = codes
    x = padded.reshape(-1,4)
    return (x[:,0] | x[:,1]<<2 | x[:,2]<<4 | x[:,3]<<6).tobytes()


class LocalRangeStore:
    def __init__(self, directory):
        self.directory=Path(directory).resolve()
        self.requests=0; self.bytes_read=0
    def get_range(self, key, offset, length):
        path=(self.directory/key).resolve()
        if self.directory not in path.parents: raise ValueError('unsafe object key')
        with path.open('rb') as f:
            f.seek(offset);data=f.read(length)
        if len(data)!=length: raise IOError('short range read')
        self.requests+=1;self.bytes_read+=length
        return data


class S3RangeStore:
    def __init__(self, client, bucket):
        self.client=client;self.bucket=bucket
        self.requests=0;self.bytes_read=0
    def get_range(self, key, offset, length):
        response=self.client.get_object(Bucket=self.bucket,Key=key,
                  Range=f'bytes={offset}-{offset+length-1}')
        body=response['Body']
        try: data=body.read()
        finally: body.close()
        if len(data)!=length: raise IOError('short S3/R2 range read')
        self.requests+=1;self.bytes_read+=length
        return data


class HttpRangeStore:
    """Read a deployed range_worker.mjs endpoint; no credentials or uploads."""
    def __init__(self,base_url,timeout=30):
        self.base_url=base_url.rstrip('/')+'/';self.timeout=timeout
        self.requests=0;self.bytes_read=0
    def get_range(self,key,offset,length):
        import re
        import urllib.request
        if not re.fullmatch(r'[a-f0-9]{64}\.tbobj',key) or offset<0 or length<=0:
            raise ValueError('invalid immutable range')
        parts=[]
        for begin in range(0,length,1<<20):
            size=min(1<<20,length-begin)
            url=f'{self.base_url}range/{key}/{offset+begin}/{size}'
            with urllib.request.urlopen(url,timeout=self.timeout) as response:data=response.read()
            if len(data)!=size:raise IOError('short HTTP range resource')
            parts.append(data);self.requests+=1;self.bytes_read+=len(data)
        return b''.join(parts)


def write_local(codes, directory, page_positions=262144, shard_bytes=1<<30):
    if page_positions<=0 or page_positions%4: raise ValueError('page_positions must be a positive multiple of four')
    if shard_bytes<=0:raise ValueError('shard_bytes must be positive')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    compressor=zstd.ZstdCompressor(level=6,write_checksum=True)
    manifest=dict(format='research-wdl2-v1',metric='unclocked-WDL',
                  codes={'draw':0,'win':1,'loss':2,'invalid':3},
                  positions=len(codes),page_positions=page_positions,frames=[])
    pending=[];payload=bytearray()
    def flush():
        if not pending:return
        key=hashlib.sha256(payload).hexdigest()+'.wdl.zst'
        path=directory/key
        if not path.exists(): path.write_bytes(payload)
        for entry in pending:entry['object']=key
        manifest['frames'].extend(pending)
        pending.clear();payload.clear()
    for begin in range(0,len(codes),page_positions):
        raw=pack(codes[begin:begin+page_positions]);compressed=compressor.compress(raw)
        if len(compressed)>shard_bytes:raise ValueError('one frame exceeds the object budget')
        if payload and len(payload)+len(compressed)>shard_bytes:flush()
        pending.append(dict(start=begin,count=min(page_positions,len(codes)-begin),
                            offset=len(payload),compressed_bytes=len(compressed),
                            raw_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        payload.extend(compressed)
    flush()
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class Reader:
    def __init__(self, manifest, store, cache_bytes=64<<20):
        self.manifest=manifest;self.store=store;self.cache_bytes=cache_bytes
        self.cache=OrderedDict();self.resident=0
        self.decoder=zstd.ZstdDecompressor()
    def frame(self, page):
        if page in self.cache:
            self.cache.move_to_end(page);return self.cache[page]
        e=self.manifest['frames'][page]
        compressed=self.store.get_range(e['object'],e['offset'],e['compressed_bytes'])
        raw=self.decoder.decompress(compressed,max_output_size=e['raw_bytes'])
        if len(raw)!=e['raw_bytes'] or hashlib.sha256(raw).hexdigest()!=e['sha256']:
            raise IOError('corrupt frame')
        if len(raw)<=self.cache_bytes:
            while self.cache and self.resident+len(raw)>self.cache_bytes:
                _,old=self.cache.popitem(last=False);self.resident-=len(old)
            self.cache[page]=raw;self.resident+=len(raw)
        return raw
    def probe_many(self, positions):
        ids=np.asarray(positions,dtype=np.int64)
        if ids.ndim!=1 or (ids<0).any() or (ids>=self.manifest['positions']).any():
            raise IndexError('position ID out of range')
        groups=defaultdict(list);size=self.manifest['page_positions']
        for i,pos in enumerate(ids):groups[int(pos)//size].append((i,int(pos)%size))
        result=np.empty(len(ids),np.uint8)
        for page,items in groups.items():
            raw=self.frame(page)
            for i,local in items:result[i]=(raw[local//4]>>((local%4)*2))&3
        return result


def validate(directory):
    rng=np.random.default_rng(487)
    codes=rng.integers(0,4,size=4096*17+3,dtype=np.uint8)
    manifest=write_local(codes,directory,page_positions=4096,shard_bytes=7000)
    store=LocalRangeStore(directory);reader=Reader(manifest,store,cache_bytes=4*1024)
    ids=rng.integers(0,len(codes),size=10000)
    actual=reader.probe_many(ids)
    assert np.array_equal(actual,codes[ids])
    assert store.requests<=len(manifest['frames']), 'batched reads should fetch each required frame once'
    # Check every stored cell, terminal partial pages and cache churn.
    assert np.array_equal(reader.probe_many(np.arange(len(codes))),codes)
    try:reader.probe_many([-1])
    except IndexError:pass
    else:raise AssertionError('invalid ID accepted')
    for n in (1,3,4,5,4095,4096,4097):
        x=rng.integers(0,4,size=n,dtype=np.uint8);raw=pack(x)
        recovered=np.array([(raw[i//4]>>((i%4)*2))&3 for i in range(n)],np.uint8)
        assert np.array_equal(x,recovered)
    first=manifest['frames'][0];path=Path(directory)/first['object']
    damaged=bytearray(path.read_bytes());damaged[first['offset']+first['compressed_bytes']//2]^=1;path.write_bytes(damaged)
    try:Reader(manifest,LocalRangeStore(directory)).probe_many([0])
    except (IOError,zstd.ZstdError):pass
    else:raise AssertionError('corruption accepted')
    return dict(status='pass',positions=len(codes),random_queries=10000,
                frames=len(manifest['frames']),objects=len({x['object'] for x in manifest['frames']}),
                requests_for_first_batch=len(set(map(lambda x:int(x)//4096,ids))),
                live_r2_tested=False,corruption_detected=True)


if __name__=='__main__':
    import argparse,tempfile
    p=argparse.ArgumentParser();p.add_argument('--output',default='r2_block_validation.json');args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='tablebase-blocks-') as directory:data=validate(directory)
    Path(args.output).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
