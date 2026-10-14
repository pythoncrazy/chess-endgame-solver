"""Validated catalog geometry and lossless JSON integers for large IDs."""
import copy
import re
import numpy as np

MAX_ID=(1<<63)-1
MAX_SAFE_JS=(1<<53)-1


def integer(value,label,minimum=0,maximum=MAX_ID):
    if isinstance(value,str):
        if not re.fullmatch(r'-?(0|[1-9][0-9]*)',value):raise ValueError(f'invalid {label}')
        value=int(value)
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or not minimum<=int(value)<=maximum:
        raise ValueError(f'invalid {label}')
    return int(value)


def normalize(manifest):
    m=copy.deepcopy(manifest)
    if m.get('format') not in ('exact-blocks-v2','exact-blocks-v3'):raise ValueError('unknown archive version')
    if not isinstance(m.get('metric'),str) or not m['metric']:raise ValueError('metric required')
    m['positions']=integer(m['positions'],'position count')
    m['page_positions']=integer(m['page_positions'],'page size',1,1<<20)
    m['index_page_records']=integer(m['index_page_records'],'index page size',1,1<<20)
    m['record_bytes']=integer(m.get('record_bytes',16),'index record bytes',16,32)
    if m['record_bytes'] not in (16,32) or (m['format']=='exact-blocks-v2' and m['record_bytes']!=16):
        raise ValueError('invalid index record version')
    if not isinstance(m['objects'],list):raise ValueError('object list required')
    if 'constant' in m:
        m['constant']=integer(m['constant'],'constant',-(1<<63),MAX_ID)
        if m['objects']:raise ValueError('constant archive must not have objects')
        return m
    previous=0
    for o in m['objects']:
        for field in ('start','count','pages','index_bytes','directory_bytes','bytes'):
            o[field]=integer(o[field],field)
        if not re.fullmatch(r'[a-f0-9]{64}\.tbobj',o['key']) or not re.fullmatch(r'[a-f0-9]{64}',o['directory_sha256']):
            raise ValueError('invalid object identity')
        if o['start']!=previous or o['count']==0:raise ValueError('object gap/overlap or empty object')
        pages=(o['count']+m['page_positions']-1)//m['page_positions']
        directories=(pages+m['index_page_records']-1)//m['index_page_records']
        if o['pages']!=pages or o['index_bytes']!=pages*m['record_bytes'] or o['directory_bytes']!=directories*32:
            raise ValueError('inconsistent index geometry')
        if o['bytes']<36+o['directory_bytes']+o['index_bytes']:raise ValueError('object too short')
        previous+=o['count']
        if previous>MAX_ID:raise ValueError('ID capacity exceeded')
    if previous!=m['positions']:raise ValueError('incomplete object coverage')
    return m


def wire(manifest):
    """JSON cannot safely transport large integers as JavaScript Numbers."""
    m=normalize(manifest)
    def walk(value):
        if isinstance(value,dict):return {k:walk(v) for k,v in value.items()}
        if isinstance(value,list):return [walk(v) for v in value]
        if isinstance(value,int) and abs(value)>MAX_SAFE_JS:return str(value)
        return value
    return walk(m)
