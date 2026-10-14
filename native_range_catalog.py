"""Preserve native Syzygy compression; authenticate physical range tiles.

Adds 32 bytes per 16 KiB source tile and a paged hash directory. Files are
content-addressed. This publisher never claims a table's chess solution was
generated or verified merely because its source bytes have been authenticated.
"""
import argparse,hashlib,json,os,tempfile,shutil
from pathlib import Path

def publish(paths,output,tile_bytes=16384,hash_page_tiles=256):
    if tile_bytes!=16384 or hash_page_tiles!=256:raise ValueError('fixed v1 geometry required')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);files=[];names=set()
    for path in map(Path,paths):
        if path.name in names or path.suffix not in ('.rtbw','.rtbz'):raise ValueError('unique native table filenames required')
        names.add(path.name);source_hash=hashlib.sha256();leaves=bytearray();size=0
        with path.open('rb') as source, tempfile.NamedTemporaryFile(dir=root,delete=False) as destination:
            temporary=Path(destination.name)
            try:
                while data:=source.read(tile_bytes):
                    destination.write(data);source_hash.update(data);leaves.extend(hashlib.sha256(data).digest());size+=len(data)
                destination.flush();os.fsync(destination.fileno())
            except BaseException:temporary.unlink(missing_ok=True);raise
        if size%64!=16:temporary.unlink(missing_ok=True);raise ValueError('invalid Syzygy size')
        key=source_hash.hexdigest()+'.tbobj';os.replace(temporary,root/key)
        index_hash=hashlib.sha256(leaves).hexdigest();index_key=index_hash+'.tbobj';index_path=root/index_key
        with tempfile.NamedTemporaryFile(dir=root,delete=False) as index:
            temp_index=Path(index.name);index.write(leaves);index.flush();os.fsync(index.fileno())
        os.replace(temp_index,index_path)
        pages=[hashlib.sha256(leaves[i:i+32*hash_page_tiles]).hexdigest() for i in range(0,len(leaves),32*hash_page_tiles)]
        files.append(dict(name=path.name,key=key,size=size,index_key=index_key,index_bytes=len(leaves),hash_pages=pages))
    catalog=dict(format='native-syzygy-ranges-v1',tile_bytes=tile_bytes,hash_page_tiles=hash_page_tiles,files=files)
    with tempfile.NamedTemporaryFile(dir=root,delete=False) as manifest:
        temp_manifest=Path(manifest.name);manifest.write((json.dumps(catalog,separators=(',',':'))+'\n').encode());manifest.flush();os.fsync(manifest.fileno())
    os.replace(temp_manifest,root/'native-catalog.json')
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return catalog

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('paths',nargs='+');a=p.parse_args()
    result=publish(a.paths,a.output);print(json.dumps(dict(files=len(result['files']),source_bytes=sum(f['size'] for f in result['files']),hash_index_bytes=sum(f['index_bytes'] for f in result['files']))))
