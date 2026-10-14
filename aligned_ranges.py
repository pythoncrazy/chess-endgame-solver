"""Stable read tiles and a containment-aware compressed-byte cache."""
from collections import OrderedDict
import bisect
from storage_manifest import integer


class RangeLRU:
    def __init__(self,budget):
        self.budget=integer(budget,'range cache budget');self.size=0
        self.items=OrderedDict();self.by_object={}
    def remove(self,token):
        raw=self.items.pop(token);self.size-=len(raw);key,offset,length=token
        rows=self.by_object[key];rows.remove((offset,length))
        if not rows:del self.by_object[key]
    def get(self,token):
        key,offset,length=token;rows=self.by_object.get(key,[])
        end=bisect.bisect_right(rows,(offset,1<<64))
        for i in range(end-1,-1,-1):
            start,size=rows[i]
            if start+size>=offset+length:
                owner=(key,start,size);raw=self.items[owner];self.items.move_to_end(owner)
                return raw[offset-start:offset-start+length]
        return None
    def put(self,token,raw):
        key,offset,length=token
        if length!=len(raw):raise ValueError('range length mismatch')
        if length>self.budget:return
        if self.get(token) is not None:return
        for start,size in list(self.by_object.get(key,[])):
            if offset<=start and start+size<=offset+length:self.remove((key,start,size))
        while self.items and self.size+length>self.budget:self.remove(next(iter(self.items)))
        self.items[token]=bytes(raw);self.size+=length
        bisect.insort(self.by_object.setdefault(key,[]),(offset,length))


class AlignedRangeStore:
    """Use stable physical byte-tile URLs, allowing reuse across query batches.

    object_sizes comes from the verified catalog. Tiles increase fetched bytes;
    enable only where locality/cache savings justify that amplification.
    """
    def __init__(self,store,object_sizes,tile_bytes=65536):
        self.store=store;self.object_sizes=object_sizes
        self.tile_bytes=integer(tile_bytes,'physical tile size',1,1<<20)
        self.requests=0;self.bytes_read=0
    def get_range(self,key,offset,length):
        if key not in self.object_sizes:raise ValueError('unknown object')
        size=self.object_sizes[key];offset=integer(offset,'range offset');length=integer(length,'range length',1)
        if offset+length>size:raise ValueError('range beyond object')
        start=offset//self.tile_bytes*self.tile_bytes;parts=[]
        for begin in range(start,offset+length,self.tile_bytes):
            count=min(self.tile_bytes,size-begin);raw=self.store.get_range(key,begin,count)
            if len(raw)!=count:raise IOError('short aligned read')
            parts.append(raw);self.requests+=1;self.bytes_read+=len(raw)
        return b''.join(parts)[offset-start:offset-start+length]
