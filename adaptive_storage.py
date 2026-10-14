"""Exact signed-32-bit blocks and compact immutable range-readable objects.

Research codec, not a chess indexer or a Syzygy-compatible prober. Encodings
are selected by actual bytes including headers; external predictions require
the identical deterministic predictor at decode time and are hash-checked.
No value rounding, no learned-only answers and no cloud mutations.
"""
from collections import OrderedDict, defaultdict
import hashlib
import json
import lzma
from pathlib import Path
import struct
import numpy as np
import zstandard as zstd
from storage_manifest import normalize,wire,integer

HEADER = struct.Struct('<4sBBBBII16s')
OBJECT_HEADER = struct.Struct('<8sIIQQI')
RECORD = struct.Struct('<QIi')  # payload offset, bytes, inline constant
RECORD3 = struct.Struct('<QIi16s')  # Adds authenticated encoded-frame checksum.
MODES = {0:'constant',1:'offset-packed',2:'palette-packed',3:'lag-residual',
         4:'predictor-sparse',5:'predictor-dense',6:'palette-planes',
         7:'raw64',8:'xor-lag64',9:'byteplanes64'}
MAX_PAGE = 1 << 20


def values32(values):
    x = np.asarray(values)
    if x.ndim != 1 or (len(x) and x.dtype.kind not in 'iu'):
        raise ValueError('one-dimensional integer values required')
    if len(x) > MAX_PAGE:
        raise ValueError('page exceeds limit')
    if len(x) and (int(x.min()) < -(1 << 31) or int(x.max()) >= (1 << 31)):
        raise ValueError('values must fit signed 32-bit without rounding')
    return x.astype(np.int64,copy=False)


def values64(values):
    x=np.asarray(values)
    if x.ndim!=1 or (len(x) and x.dtype.kind not in 'iu') or len(x)>MAX_PAGE:
        raise ValueError('one-dimensional integer page required')
    if len(x) and (int(x.min())<-(1<<63) or int(x.max())>=(1<<63)):
        raise ValueError('values must fit signed 64-bit without rounding')
    return x.astype(np.int64,copy=False)


def digest(x,wide=False):
    return hashlib.blake2b(np.asarray(x,dtype='<i8' if wide else '<i4').tobytes(),digest_size=16).digest()


def pack_unsigned(x, width, planes=False):
    if not width: return b''
    x=np.asarray(x,dtype=np.uint64)
    if not planes and width in (1,2,4,8):
        cells=8//width;padded=np.zeros(((len(x)+cells-1)//cells)*cells,np.uint8)
        padded[:len(x)]=x
        return np.bitwise_or.reduce(padded.reshape(-1,cells)<<np.arange(0,8,width,dtype=np.uint8),axis=1).tobytes()
    bits=((x[:,None] >> np.arange(width,dtype=np.uint64)) & 1).astype(np.uint8)
    if planes:
        return np.packbits(bits.T,axis=1,bitorder='little').tobytes()
    return np.packbits(bits.reshape(-1),bitorder='little').tobytes()


def unpack_unsigned(raw,n,width,planes=False):
    if width > 33: raise ValueError('invalid bit width')
    if not width:
        if raw: raise ValueError('unexpected zero-width payload')
        return np.zeros(n,np.uint64)
    expected=width*((n+7)//8) if planes else (n*width+7)//8
    if len(raw)!=expected: raise ValueError('bit payload length mismatch')
    if planes and n%8:
        tails=np.frombuffer(raw,np.uint8).reshape(width,-1)[:,-1]
        if (tails>>(n%8)).any():raise ValueError('nonzero plane padding')
    elif not planes and (n*width)%8 and raw[-1]>>((n*width)%8):
        raise ValueError('nonzero packed padding')
    if not planes and width in (1,2,4,8):
        b=np.frombuffer(raw,dtype=np.uint8);cells=8//width;out=np.empty(len(b)*cells,np.uint8)
        for j in range(cells):out[j::cells]=(b>>(j*width))&((1<<width)-1)
        return out[:n]
    bits=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little')
    if planes:bits=bits.reshape(width,-1)[:,:n].T
    else:bits=bits[:n*width].reshape(n,width)
    return (bits.astype(np.uint64)<<np.arange(width,dtype=np.uint64)).sum(axis=1)


def zigzag(x):
    x=np.asarray(x,dtype=np.int64)
    return np.where(x>=0,x*2,-x*2-1).astype(np.uint64)


def unzigzag(x):
    x=np.asarray(x,dtype=np.uint64)
    return (x>>1).astype(np.int64) * np.where(x&1,-1,1) - (x&1).astype(np.int64)


def varint(x):
    out=bytearray()
    while x>=128:out.append((x&127)|128);x>>=7
    out.append(x);return out


def take_varint(raw,offset):
    result=0
    for shift in range(0,42,7):
        if offset>=len(raw): raise ValueError('truncated varint')
        b=raw[offset];offset+=1;result|=(b&127)<<shift
        if b<128:return result,offset
    raise ValueError('oversized varint')


def candidates(values,prediction=None,level=12,allow_lzma=True):
    x=values64(values);n=len(x);wide=bool(n and (int(x.min())<-(1<<31) or int(x.max())>=(1<<31)))
    check=digest(x,wide);out=[]
    def add(mode,width,body):
        choices=[(0,body),(1,zstd.ZstdCompressor(level=level).compress(body))]
        if allow_lzma and body:
            choices.append((2,lzma.compress(body,preset=6)))
        for comp,payload in choices:
            frame=HEADER.pack(b'TBC2' if wide else b'TBC1',mode,comp,width,0,n,len(payload),check)+payload
            out.append((frame,dict(mode=MODES[mode],compression=('raw','zstd','lzma')[comp],bytes=len(frame))))
    if n==0 or np.all(x==x[0]):
        add(0,0,struct.pack('<q' if wide else '<i',int(x[0]) if n else 0))
    if wide:
        if prediction is not None:raise ValueError('wide external predictors are not implemented')
        raw=x.astype('<i8',copy=False).tobytes()
        add(7,64,raw)
        add(9,64,np.frombuffer(raw,np.uint8).reshape(-1,8).T.tobytes())
        for lag in (1,8,64):
            if lag>n:continue
            u=x.astype('<i8',copy=False).view('<u8');residual=u.copy();residual[lag:]^=u[:-lag]
            add(8,64,struct.pack('<I',lag)+residual.tobytes())
        return out
    if n:
        base=int(x.min());u=(x-base).astype(np.uint64)
        width=int(u.max()).bit_length()
        add(1,width,struct.pack('<i',base)+pack_unsigned(u,width))
        palette,ranks=np.unique(x,return_inverse=True)
        if len(palette)<=256:
            width=(len(palette)-1).bit_length()
            prefix=struct.pack('<H',len(palette))+palette.astype('<i4').tobytes()
            add(2,width,prefix+pack_unsigned(ranks,width))
            add(6,width,prefix+pack_unsigned(ranks,width,True))
        for lag in (1,8,64):
            if lag>n:continue
            residual=x.copy();residual[lag:]-=x[:-lag]
            u=zigzag(residual);width=int(u.max()).bit_length()
            add(3,width,struct.pack('<I',lag)+pack_unsigned(u,width,True))
        if prediction is not None:
            pred=values32(prediction)
            if len(pred)!=n:raise ValueError('predictor length mismatch')
            residual=x-pred;u=zigzag(residual);width=int(u.max()).bit_length()
            add(5,width,digest(pred)+pack_unsigned(u,width,True))
            ids=np.flatnonzero(residual);body=bytearray(digest(pred)+struct.pack('<I',len(ids)))
            previous=-1
            for i in ids:
                body.extend(varint(int(i)-previous-1));body.extend(varint(int(u[i])));previous=int(i)
            add(4,0,bytes(body))
    return out


def encode(values,prediction=None,level=12,allow_lzma=True):
    result=min(candidates(values,prediction,level,allow_lzma),key=lambda c:len(c[0]))
    if not np.array_equal(decode(result[0],prediction),values64(values)):raise ValueError('encoder round-trip failure')
    return result


def frame_parts(frame):
    if len(frame)<HEADER.size:raise ValueError('truncated header')
    magic,mode,comp,width,reserved,n,size,check=HEADER.unpack_from(frame)
    wide=magic==b'TBC2'
    if magic not in (b'TBC1',b'TBC2') or reserved or mode not in MODES or comp>2 or n>MAX_PAGE:
        raise ValueError('invalid header')
    if (wide and mode not in (0,7,8,9)) or (not wide and mode>6):raise ValueError('mode/version mismatch')
    if n==0 and (mode!=0 or wide):raise ValueError('noncanonical empty frame')
    if mode in (0,4) and width!=0:raise ValueError('noncanonical zero-width mode')
    if (wide and mode!=0 and width!=64) or (not wide and width>33):raise ValueError('invalid bit width')
    if len(frame)!=HEADER.size+size:raise ValueError('frame length mismatch')
    payload=frame[HEADER.size:];limit=16*n+4096
    if comp==1:
        declared=zstd.frame_content_size(payload)
        if declared not in (zstd.CONTENTSIZE_UNKNOWN,zstd.CONTENTSIZE_ERROR) and declared>limit:
            raise ValueError('oversized decompressed payload')
        body=zstd.ZstdDecompressor().decompress(payload,max_output_size=limit,allow_extra_data=False)
    elif comp==2:
        dec=lzma.LZMADecompressor(memlimit=64<<20);body=dec.decompress(payload,max_length=limit+1)
        if not dec.eof or dec.unused_data:raise ValueError('invalid LZMA frame')
    else:body=payload
    if len(body)>limit:raise ValueError('oversized payload')
    return n,mode,width,wide,body,check


def decode(frame,prediction=None):
    n,mode,width,wide,body,check=frame_parts(frame)
    if mode==0:
        if len(body)!=(8 if wide else 4):raise ValueError('invalid constant')
        value=struct.unpack('<q' if wide else '<i',body)[0]
        if n==0 and value!=0:raise ValueError('noncanonical empty value')
        x=np.full(n,value,np.int64)
    elif wide:
        if mode in (7,9):
            if len(body)!=8*n:raise ValueError('invalid wide payload length')
            raw=body if mode==7 else np.frombuffer(body,np.uint8).reshape(8,n).T.copy().tobytes()
            x=np.frombuffer(raw,dtype='<i8')
        else:
            if len(body)!=4+8*n:raise ValueError('invalid wide residual length')
            lag=struct.unpack_from('<I',body)[0]
            if not 1<=lag<=n:raise ValueError('invalid lag')
            r=np.frombuffer(body[4:],dtype='<u8');padded=np.zeros(((n+lag-1)//lag)*lag,np.uint64);padded[:n]=r
            x=np.bitwise_xor.accumulate(padded.reshape(-1,lag),axis=0).reshape(-1)[:n].view(np.int64)
    elif mode==1:
        if len(body)<4:raise ValueError('missing base')
        x=unpack_unsigned(body[4:],n,width).astype(np.int64)+struct.unpack_from('<i',body)[0]
    elif mode in (2,6):
        if len(body)<2:raise ValueError('missing palette')
        count=struct.unpack_from('<H',body)[0];end=2+4*count
        if not 1<=count<=256 or len(body)<end:raise ValueError('invalid palette')
        palette=np.frombuffer(body[2:end],dtype='<i4').astype(np.int64)
        ranks=unpack_unsigned(body[end:],n,width,mode==6)
        if (ranks>=count).any():raise ValueError('invalid rank')
        x=palette[ranks]
    elif mode==3:
        if len(body)<4:raise ValueError('missing lag')
        lag=struct.unpack_from('<I',body)[0]
        if not 1<=lag<=n:raise ValueError('invalid lag')
        r=unzigzag(unpack_unsigned(body[4:],n,width,True))
        padded=np.zeros(((n+lag-1)//lag)*lag,np.int64);padded[:n]=r
        x=np.cumsum(padded.reshape(-1,lag),axis=0).reshape(-1)[:n]
    else:
        if prediction is None:raise ValueError('predictor required')
        pred=values32(prediction)
        if len(pred)!=n or len(body)<16 or body[:16]!=digest(pred):
            raise ValueError('predictor identity mismatch')
        if mode==5:
            x=pred+unzigzag(unpack_unsigned(body[16:],n,width,True))
        else:
            if len(body)<20:raise ValueError('missing exception count')
            count=struct.unpack_from('<I',body,16)[0]
            if count>n:raise ValueError('too many exceptions')
            x=pred.copy();offset=20;previous=-1
            for _ in range(count):
                gap,offset=take_varint(body,offset);u,offset=take_varint(body,offset)
                i=previous+gap+1
                if i>=n:raise ValueError('exception index out of range')
                x[i]+=-(u//2)-1 if u&1 else u//2;previous=i
            if offset!=len(body):raise ValueError('trailing exception bytes')
    if len(x)!=n or digest(x,wide)!=check:raise ValueError('decoded checksum mismatch')
    return values64(x) if wide else values32(x)


def commit_manifest(manifest,directory):
    """Replace the local root only after all immutable objects are complete."""
    import os,tempfile
    data=(json.dumps(wire(manifest),separators=(',',':'))+'\n').encode()
    with tempfile.NamedTemporaryFile(dir=directory,prefix='manifest-',delete=False) as file:
        temporary=Path(file.name)
        try:file.write(data);file.flush();os.fsync(file.fileno())
        except BaseException:temporary.unlink(missing_ok=True);raise
    try:os.replace(temporary,Path(directory)/'manifest.json')
    finally:temporary.unlink(missing_ok=True)
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_archive(values,directory,page_positions=65536,object_bytes=1<<30,metric='integer',
                  level=12,allow_lzma=True,index_page_records=2048,max_pages_per_object=65536,catalog_fields=None):
    x=np.asarray(values)
    if x.ndim!=1 or (len(x) and x.dtype.kind not in 'iu'):raise ValueError('one-dimensional integer values required')
    page_positions=integer(page_positions,'page size',1,MAX_PAGE)
    return write_archive_pages((x[b:b+page_positions] for b in range(0,len(x),page_positions)),directory,
        page_positions,object_bytes,metric,level,allow_lzma,index_page_records,max_pages_per_object,catalog_fields)


def write_archive_pages(pages,directory,page_positions=65536,object_bytes=1<<30,metric='integer',
                        level=12,allow_lzma=True,index_page_records=2048,max_pages_per_object=65536,catalog_fields=None):
    """One-pass streaming writer; at most 16 MiB of payload spool stays in RAM.

    Except the last page, every input page must have page_positions cells.
    Local root replacement is atomic; distributed ownership/CAS is separate.
    """
    import os,tempfile
    page_positions=integer(page_positions,'page size',1,MAX_PAGE)
    index_page_records=integer(index_page_records,'index page records',1,MAX_PAGE)
    object_bytes=integer(object_bytes,'object byte limit',OBJECT_HEADER.size+32+RECORD3.size)
    max_pages_per_object=integer(max_pages_per_object,'pages per object',1,1<<20)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    manifest=dict(format='exact-blocks-v3',metric=metric,positions=0,page_positions=page_positions,
                  index_page_records=index_page_records,record_bytes=RECORD3.size,objects=[])
    if catalog_fields:
        if set(catalog_fields)&set(manifest):raise ValueError('catalog fields override geometry')
        manifest.update(catalog_fields)
    records=[];start=0;covered=0;payload_bytes=0;partial_seen=False
    payload=tempfile.SpooledTemporaryFile(max_size=16<<20)
    def flush():
        nonlocal records,start,covered,payload_bytes
        if not records:return
        header=OBJECT_HEADER.pack(b'TBOBJ003',3,page_positions,start,covered,len(records))
        directory_bytes=32*((len(records)+index_page_records-1)//index_page_records)
        base=len(header)+directory_bytes+len(records)*RECORD3.size;offset=base;index=bytearray()
        for length,value,checksum in records:
            index.extend(RECORD3.pack(offset if length else 0,length,value,checksum));offset+=length
        chunk_size=index_page_records*RECORD3.size
        hash_directory=b''.join(hashlib.sha256(index[k:k+chunk_size]).digest() for k in range(0,len(index),chunk_size))
        digestor=hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=directory,prefix='object-',delete=False) as output:
            temporary=Path(output.name)
            try:
                for data in (header,hash_directory,index):output.write(data);digestor.update(data)
                payload.seek(0)
                while True:
                    data=payload.read(1<<20)
                    if not data:break
                    output.write(data);digestor.update(data)
                output.flush();os.fsync(output.fileno())
            except BaseException:temporary.unlink(missing_ok=True);raise
        key=digestor.hexdigest()+'.tbobj'
        try:os.replace(temporary,directory/key)
        finally:temporary.unlink(missing_ok=True)
        manifest['objects'].append(dict(key=key,start=start,count=covered,pages=len(records),
                    index_bytes=len(index),directory_bytes=len(hash_directory),
                    directory_sha256=hashlib.sha256(header+hash_directory).hexdigest(),bytes=offset))
        start+=covered;records=[];covered=0;payload_bytes=0;payload.seek(0);payload.truncate(0)
    try:
        for incoming in pages:
            page=values64(incoming)
            if not len(page) or len(page)>page_positions or partial_seen:
                raise ValueError('nonfinal short page, empty page or oversized page')
            partial_seen=len(page)<page_positions
            if np.all(page==page[0]) and -(1<<31)<=int(page[0])<(1<<31):value=int(page[0]);frame=b''
            else:value=0;frame,_=encode(page,level=level,allow_lzma=allow_lzma)
            next_records=len(records)+1
            required=OBJECT_HEADER.size+32*((next_records+index_page_records-1)//index_page_records)+next_records*RECORD3.size+payload_bytes+len(frame)
            if records and (required>object_bytes or len(records)>=max_pages_per_object):flush()
            if OBJECT_HEADER.size+32+RECORD3.size+len(frame)>object_bytes:raise ValueError('frame exceeds object budget')
            checksum=hashlib.blake2b(frame,digest_size=16).digest() if frame else bytes(16)
            records.append((len(frame),value,checksum));payload.write(frame);covered+=len(page);payload_bytes+=len(frame)
            manifest['positions']+=len(page)
        flush();commit_manifest(manifest,directory)
    finally:payload.close()
    return manifest


def write_constant_archive(value,positions,directory,metric='integer'):
    value=integer(value,'constant',-(1<<63));positions=integer(positions,'position count')
    manifest=dict(format='exact-blocks-v3',metric=metric,positions=positions,
                  page_positions=65536,index_page_records=2048,constant=value,objects=[])
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    commit_manifest(manifest,directory)
    return manifest


class ArchiveReader:
    def __init__(self,manifest,store,cache_bytes=64<<20):
        manifest=normalize(manifest);self.manifest=manifest;self.store=store;self.cache_bytes=cache_bytes
        self.index={};self.cache=OrderedDict();self.resident=0
        self.starts=np.array([o['start'] for o in manifest['objects']],np.int64)
        self.record=RECORD3 if manifest['record_bytes']==32 else RECORD
    def object_index(self,j):
        if j in self.index:return self.index[j]
        o=self.manifest['objects'][j];directory_end=OBJECT_HEADER.size+o['directory_bytes'];size=directory_end+o['index_bytes']
        raw=self.store.get_range(o['key'],0,size)
        if hashlib.sha256(raw[:directory_end]).hexdigest()!=o['directory_sha256']:raise ValueError('object directory checksum mismatch')
        magic,version,page,start,count,npages=OBJECT_HEADER.unpack_from(raw)
        expected_magic,expected_version=(b'TBOBJ003',3) if self.record==RECORD3 else (b'TBOBJ002',2)
        if (magic,version,page,start,count,npages)!=(expected_magic,expected_version,self.manifest['page_positions'],o['start'],o['count'],o['pages']):
            raise ValueError('object metadata mismatch')
        chunk_size=self.manifest['index_page_records']*self.record.size
        for k in range(0,o['index_bytes'],chunk_size):
            expected=raw[OBJECT_HEADER.size+32*(k//chunk_size):OBJECT_HEADER.size+32*(k//chunk_size+1)]
            if hashlib.sha256(raw[directory_end+k:directory_end+min(k+chunk_size,o['index_bytes'])]).digest()!=expected:
                raise ValueError('object index checksum mismatch')
        records=[self.record.unpack_from(raw,directory_end+i*self.record.size) for i in range(npages)]
        self.index[j]=records;return records
    def get_page(self,j,page):
        key=(j,page)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        entry=self.object_index(j)[page];offset,length,value=entry[:3]
        o=self.manifest['objects'][j];n=min(self.manifest['page_positions'],o['count']-page*self.manifest['page_positions'])
        if not length:return np.full(n,value,np.int64)
        if offset<OBJECT_HEADER.size+o['directory_bytes']+o['index_bytes'] or offset+length>o['bytes']:
            raise ValueError('invalid frame range')
        raw=self.store.get_range(o['key'],offset,length)
        if len(entry)==4 and hashlib.blake2b(raw,digest_size=16).digest()!=entry[3]:raise ValueError('encoded frame checksum mismatch')
        values=decode(raw)
        if len(values)!=n:raise ValueError('page count mismatch')
        if values.nbytes<=self.cache_bytes:
            while self.cache and self.resident+values.nbytes>self.cache_bytes:
                _,old=self.cache.popitem(last=False);self.resident-=old.nbytes
            self.cache[key]=values;self.resident+=values.nbytes
        return values
    def probe_many(self,positions):
        ids=np.asarray(positions)
        if ids.ndim!=1 or (len(ids) and ids.dtype.kind not in 'iu'):raise IndexError('integer position IDs required')
        if len(ids) and (int(ids.min())<0 or int(ids.max())>=self.manifest['positions']):raise IndexError('ID out of range')
        if 'constant' in self.manifest:return np.full(len(ids),self.manifest['constant'],np.int64)
        ids=ids.astype(np.int64);owners=np.searchsorted(self.starts,ids,side='right')-1
        groups=defaultdict(list);size=self.manifest['page_positions'];result=np.empty(len(ids),np.int64)
        for k,(pos,j) in enumerate(zip(ids,owners)):
            local=int(pos)-self.manifest['objects'][int(j)]['start']
            groups[(int(j),local//size)].append((k,local%size))
        for (j,page),items in groups.items():
            values=self.get_page(j,page)
            for k,local in items:result[k]=values[local]
        return result
