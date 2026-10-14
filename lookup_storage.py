"""Paged authenticated indexes, bounded caches and coalesced range queries.

Synchronous batch API for one reader. Optional bounded parallel range fetches
must have a thread-safe store. Concurrent callers should use separate readers or
serialize calls. A shared production service must also have single-flight logic.
"""
from collections import OrderedDict,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import struct
import threading
import numpy as np
from adaptive_storage import OBJECT_HEADER,RECORD,RECORD3,decode
from storage_manifest import normalize,integer


class ByteLRU:
    def __init__(self,budget):
        if not isinstance(budget,(int,np.integer)) or isinstance(budget,bool) or budget<0:
            raise ValueError('cache budget must be a nonnegative integer')
        self.budget=budget;self.size=0;self.items=OrderedDict()
        self.lock=threading.RLock()
    def get(self,key):
        with self.lock:
            if key not in self.items:return None
            self.items.move_to_end(key);return self.items[key]
    def put(self,key,value):
        with self.lock:
            size=value.nbytes if hasattr(value,'nbytes') else len(value)
            if size>self.budget:return
            if key in self.items:
                old=self.items.pop(key);self.size-=old.nbytes if hasattr(old,'nbytes') else len(old)
            while self.items and self.size+size>self.budget:
                _,old=self.items.popitem(last=False);self.size-=old.nbytes if hasattr(old,'nbytes') else len(old)
            self.items[key]=value;self.size+=size


def compact(values):
    if not len(values):return values.astype(np.int8)
    lo,hi=int(values.min()),int(values.max())
    dtype=np.int8 if -128<=lo and hi<=127 else np.int16 if -32768<=lo and hi<=32767 else np.int32 if -(1<<31)<=lo and hi<(1<<31) else np.int64
    result=values.astype(dtype);result.setflags(write=False);return result


def coalesce(ranges,max_gap=4096,max_bytes=1<<20,max_amplification=2.0):
    """Return contiguous GET intervals and slices, never a multipart Range."""
    if max_gap<0 or max_bytes<=0 or max_amplification<1:raise ValueError('invalid range policy')
    grouped=[];pieces=[]
    for offset,length,tag in ranges:
        if offset<0 or length<=0:raise ValueError('invalid range')
        for begin in range(0,length,max_bytes):pieces.append((offset+begin,min(length-begin,max_bytes),tag))
    for offset,length,tag in sorted(pieces,key=lambda r:r[0]):
        if grouped:
            old=grouped[-1];end=max(old['end'],offset+length);useful=old['useful']+length
            if offset-old['end']<=max_gap and end-old['start']<=max_bytes and end-old['start']<=max_amplification*useful:
                old['end']=end;old['useful']=useful;old['items'].append((offset,length,tag));continue
        grouped.append(dict(start=offset,end=offset+length,useful=length,items=[(offset,length,tag)]))
    return grouped


class BatchReader:
    def __init__(self,manifest,store,data_cache_bytes=64<<20,index_cache_bytes=8<<20,
                 directory_cache_bytes=4<<20,max_gap=4096,max_bytes=1<<20,
                 max_amplification=2.0,max_parallel_reads=1,max_batch_decoded_bytes=64<<20):
        manifest=normalize(manifest)
        self.record=RECORD3 if manifest['record_bytes']==32 else RECORD
        self.manifest=manifest;self.store=store
        self.data=ByteLRU(data_cache_bytes);self.index=ByteLRU(index_cache_bytes);self.directory=ByteLRU(directory_cache_bytes)
        self.starts=np.asarray([o['start'] for o in manifest['objects']],np.int64)
        self.max_gap=integer(max_gap,'range gap');self.max_bytes=integer(max_bytes,'range cap',1)
        if not np.isfinite(max_amplification) or max_amplification<1:raise ValueError('invalid amplification limit')
        self.max_amplification=max_amplification
        self.max_parallel_reads=integer(max_parallel_reads,'parallel reads',1,256)
        self.max_batch_decoded_bytes=integer(max_batch_decoded_bytes,'decode working budget',1)
        self.stats=dict(decoded_pages=0,packed_pages=0,data_cache_hits=0,inline_pages=0,range_groups=0,useful_range_bytes=0,
                        max_staged_decoded_bytes=0)
    def cache_tag(self,j,page=None):
        key=self.manifest['objects'][j]['key']
        return key if page is None else (key,page)
    def windows(self,j,pages):
        window={};working=0;o=self.manifest['objects'][j];size=self.manifest['page_positions']
        for page,items in pages.items():
            cost=min(size,o['count']-page*size)*8
            if window and working+cost>self.max_batch_decoded_bytes:
                self.stats['max_staged_decoded_bytes']=max(self.stats['max_staged_decoded_bytes'],working)
                yield window;window={};working=0
            window[page]=items;working+=cost
        if window:
            self.stats['max_staged_decoded_bytes']=max(self.stats['max_staged_decoded_bytes'],working)
            yield window
    def peek(self,position):
        """Return an already-decoded cell, or None, without any storage read."""
        if not isinstance(position,(int,np.integer)) or isinstance(position,bool) or not 0<=position<self.manifest['positions']:
            raise IndexError('integer ID out of range')
        if 'constant' in self.manifest:return self.manifest['constant']
        j=int(np.searchsorted(self.starts,position,side='right')-1)
        local=position-self.manifest['objects'][j]['start'];page=local//self.manifest['page_positions']
        cached=self.data.get(self.cache_tag(j,page))
        return None if cached is None else int(cached[local%self.manifest['page_positions']])
    def fetch(self,j,ranges):
        key=self.manifest['objects'][j]['key']
        groups=coalesce(ranges,self.max_gap,self.max_bytes,self.max_amplification)
        def read(g):return self.store.get_range(key,g['start'],g['end']-g['start'])
        if self.max_parallel_reads>1 and len(groups)>1:
            with ThreadPoolExecutor(max_workers=self.max_parallel_reads) as pool:payloads=list(pool.map(read,groups))
        else:payloads=[read(g) for g in groups]
        pieces=defaultdict(list)
        for g,raw in zip(groups,payloads):
            if len(raw)!=g['end']-g['start']:raise IOError('short range read')
            for offset,length,tag in g['items']:pieces[tag].append((offset,raw[offset-g['start']:offset-g['start']+length]))
            self.stats['range_groups']+=1;self.stats['useful_range_bytes']+=g['useful']
        return {tag:b''.join(raw for _,raw in sorted(parts)) for tag,parts in pieces.items()}
    def get_directory(self,j):
        cached=self.directory.get(self.cache_tag(j))
        if cached is not None:return cached
        o=self.manifest['objects'][j];n=OBJECT_HEADER.size+o['directory_bytes']
        raw=self.store.get_range(o['key'],0,n)
        if hashlib.sha256(raw).hexdigest()!=o['directory_sha256']:raise ValueError('directory checksum mismatch')
        meta=OBJECT_HEADER.unpack_from(raw)
        magic,version=(b'TBOBJ003',3) if self.record==RECORD3 else (b'TBOBJ002',2)
        expected=(magic,version,self.manifest['page_positions'],o['start'],o['count'],o['pages'])
        if meta!=expected:raise ValueError('object metadata mismatch')
        result=raw[OBJECT_HEADER.size:];self.directory.put(self.cache_tag(j),result);return result
    def records(self,j,pages):
        o=self.manifest['objects'][j];directory=self.get_directory(j);count=self.manifest['index_page_records']
        chunks=sorted({p//count for p in pages});data={};missing=[]
        for chunk in chunks:
            raw=self.index.get(self.cache_tag(j,chunk))
            if raw is not None:data[chunk]=raw;continue
            offset=chunk*count*self.record.size;length=min(count*self.record.size,o['index_bytes']-offset)
            missing.append((OBJECT_HEADER.size+o['directory_bytes']+offset,length,chunk))
        fetched=self.fetch(j,missing)
        for chunk,raw in fetched.items():
            if hashlib.sha256(raw).digest()!=directory[chunk*32:chunk*32+32]:raise ValueError('index checksum mismatch')
            self.index.put(self.cache_tag(j,chunk),raw);data[chunk]=raw
        return {p:self.record.unpack_from(data[p//count],(p%count)*self.record.size) for p in pages}
    def probe_many(self,positions):
        ids=np.asarray(positions)
        if ids.ndim!=1 or (len(ids) and ids.dtype.kind not in 'iu'):raise IndexError('integer IDs required')
        if len(ids) and (int(ids.min())<0 or int(ids.max())>=self.manifest['positions']):raise IndexError('ID out of range')
        if 'constant' in self.manifest:return np.full(len(ids),self.manifest['constant'],np.int64)
        ids=ids.astype(np.int64);owners=np.searchsorted(self.starts,ids,side='right')-1
        grouped=defaultdict(lambda:defaultdict(list));size=self.manifest['page_positions'];out=np.empty(len(ids),np.int64)
        for k,(pos,j) in enumerate(zip(ids,owners)):
            local=int(pos)-self.manifest['objects'][int(j)]['start']
            grouped[int(j)][local//size].append((k,local%size))
        for j,pages in ((j,window) for j,all_pages in grouped.items() for window in self.windows(j,all_pages)):
            values={};missing=[]
            for page in pages:
                cached=self.data.get(self.cache_tag(j,page))
                if cached is not None:values[page]=cached;self.stats['data_cache_hits']+=1
                else:missing.append(page)
            if missing:
                records=self.records(j,missing);ranges=[];o=self.manifest['objects'][j]
                for page,entry in records.items():
                    offset,length,value=entry[:3]
                    n=min(size,o['count']-page*size)
                    if not length:
                        values[page]=compact(np.full(n,value,np.int64));self.stats['inline_pages']+=1
                    else:
                        if offset<OBJECT_HEADER.size+o['directory_bytes']+o['index_bytes'] or offset+length>o['bytes']:
                            raise ValueError('invalid data range')
                        ranges.append((offset,length,page))
                fetched=self.fetch(j,ranges)
                for page,raw in fetched.items():
                    from packed_lookup import prepare
                    x=prepare(raw,records[page][3]) if len(records[page])==4 else None
                    if x is None:
                        x=compact(decode(raw));self.stats['decoded_pages']+=1
                    else:self.stats['packed_pages']+=1
                    n=min(size,o['count']-page*size)
                    if len(x)!=n:raise ValueError('page count mismatch')
                    values[page]=x
                for page in missing:self.data.put(self.cache_tag(j,page),values[page])
            for page,items in pages.items():
                x=values[page]
                locations=np.fromiter((local for _,local in items),np.int64,count=len(items))
                targets=np.fromiter((k for k,_ in items),np.int64,count=len(items))
                out[targets]=x.take(locations) if hasattr(x,'take') and not isinstance(x,np.ndarray) else x[locations]
        return out
