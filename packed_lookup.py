"""Selective cell access to authenticated packed frames, without full unpack.

Only v3 index checksums authorize this path. The writer must verify each full
decoded page against its source values before it commits that checksum.
Legacy v2 frames keep the full checksum verification of decoded values at lookup.
"""
import hashlib
import struct
import numpy as np
from adaptive_storage import frame_parts


class PackedPage:
    def __init__(self,n,mode,width,wide,body):
        self.n=n;self.mode=mode;self.width=width;self.wide=wide;self.body=body
        self.nbytes=len(body)+64
        if mode==0:
            if len(body)!=(8 if wide else 4):raise ValueError('invalid constant')
            self.value=struct.unpack('<q' if wide else '<i',body)[0]
        elif mode==1:
            if len(body)<4:raise ValueError('missing base')
            self.base=struct.unpack_from('<i',body)[0];self.bits=memoryview(body)[4:]
            if len(self.bits)!=(n*width+7)//8:raise ValueError('invalid packed length')
        elif mode in (2,6):
            if len(body)<2:raise ValueError('missing palette')
            count=struct.unpack_from('<H',body)[0];end=2+4*count
            if not 1<=count<=256 or len(body)<end:raise ValueError('invalid palette')
            self.palette=np.frombuffer(memoryview(body)[2:end],dtype='<i4');self.bits=memoryview(body)[end:]
            expected=width*((n+7)//8) if mode==6 else (n*width+7)//8
            if len(self.bits)!=expected:raise ValueError('invalid palette payload')
        elif mode in (7,9):
            if len(body)!=8*n:raise ValueError('invalid wide payload')
            if mode==7:self.raw=np.frombuffer(body,dtype='<i8')
        else:raise ValueError('mode needs full reconstruction')
    def __len__(self):return self.n
    def code(self,i):
        if not self.width:return 0
        if self.mode==6:
            stride=(self.n+7)//8
            return sum(((self.bits[b*stride+i//8]>>(i%8))&1)<<b for b in range(self.width))
        begin=i*self.width;byte=begin//8;shift=begin%8;count=(shift+self.width+7)//8
        value=int.from_bytes(self.bits[byte:byte+count],'little')
        return (value>>shift)&((1<<self.width)-1)
    def __getitem__(self,i):
        if isinstance(i,slice):return self.take(np.arange(self.n)[i])
        if not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)):raise IndexError('integer local ID required')
        i=int(i)
        if not 0<=i<self.n:raise IndexError('cell out of range')
        if self.mode==0:return self.value
        if self.mode==1:return self.base+self.code(i)
        if self.mode in (2,6):
            code=self.code(i)
            if code>=len(self.palette):raise ValueError('invalid palette rank')
            return int(self.palette[code])
        if self.mode==7:return int(self.raw[i])
        value=sum(self.body[b*self.n+i]<<(8*b) for b in range(8))
        return value-(1<<64) if value>=(1<<63) else value
    def take(self,indices):
        ids=np.asarray(indices)
        if ids.ndim!=1 or (len(ids) and ids.dtype.kind not in 'iu'):raise IndexError('integer local IDs required')
        if not len(ids):return np.empty(0,np.int64)
        if int(ids.min())<0 or int(ids.max())>=self.n:raise IndexError('cell out of range')
        ids=ids.astype(np.int64,copy=False)
        if self.mode==0:return np.full(len(ids),self.value,np.int64)
        if self.mode==7:return self.raw[ids].astype(np.int64,copy=False)
        if self.mode==9:
            source=np.frombuffer(self.body,np.uint8);value=np.zeros(len(ids),np.uint64)
            for b in range(8):value|=source[b*self.n+ids].astype(np.uint64)<<(8*b)
            return value.view(np.int64)
        width=self.width;rank=np.zeros(len(ids),np.uint64)
        source=np.frombuffer(self.bits,np.uint8)
        if width and self.mode==6:
            stride=(self.n+7)//8;byte=ids//8;shift=(ids%8).astype(np.uint64)
            for b in range(width):rank|=((source[b*stride+byte].astype(np.uint64)>>shift)&1)<<b
        elif width:
            begin=ids*width;byte=begin//8;shift=(begin%8).astype(np.uint64)
            padded=np.concatenate([source,np.zeros(5,np.uint8)])
            for b in range((width+14)//8):rank|=padded[byte+b].astype(np.uint64)<<(8*b)
            rank=(rank>>shift)&((1<<width)-1)
        if self.mode==1:return rank.astype(np.int64)+self.base
        if (rank>=len(self.palette)).any():raise ValueError('invalid palette rank')
        return self.palette[rank].astype(np.int64)


def prepare(frame,encoded_checksum):
    if hashlib.blake2b(frame,digest_size=16).digest()!=encoded_checksum:
        raise ValueError('encoded frame checksum mismatch')
    n,mode,width,wide,body,_=frame_parts(frame)
    if mode not in (0,1,2,6,7,9):return None
    return PackedPage(n,mode,width,wide,body)
