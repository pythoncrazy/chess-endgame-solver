"""Generate independent codec/archive/graph fixtures for the Rust implementation."""
import hashlib,json
from pathlib import Path
import numpy as np
from adaptive_storage import candidates,write_archive
from lookup_storage import BatchReader
from r2_blocks import LocalRangeStore

def main():
    root=Path(__file__).parent/'rust'/'fixtures';root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(93711)
    arrays=[np.array([],np.int64),np.array([0],np.int64),np.full(193,-2,np.int64),
            rng.integers(-2,3,193,dtype=np.int64),np.arange(193,dtype=np.int64)-96,
            np.array([-(1<<31),(1<<31)-1]*97,np.int64),
            np.array([-(1<<63),(1<<63)-1,0,-1,1]*39,np.int64)]
    arrays += [rng.integers(-2,3,n,dtype=np.int64) for n in [7,8,9,31,32,33,63,64,65,255,256,257]]
    fixtures=[]
    for values in arrays:
        prediction=np.zeros(len(values),np.int64) if len(values) and values.min()>=-(1<<31) and values.max()<(1<<31) else None
        for frame,meta in candidates(values,prediction):
            fixtures.append(dict(frame=frame.hex(),hash=hashlib.blake2b(frame,digest_size=16).hexdigest(),values=values.tolist(),prediction=None if prediction is None else prediction.tolist(),**meta))
    (root/'frames.json').write_text(json.dumps(fixtures,separators=(',',':')))
    archive_values=[]
    for name,values in [('mixed',np.concatenate([np.zeros(65,np.int64),rng.integers(-2,3,65,dtype=np.int64),np.full(65,1<<55,np.int64),rng.integers(-(1<<60),1<<60,193,dtype=np.int64)])),
                        ('second',rng.integers(-2,3,388,dtype=np.int64))]:
        write_archive(values,root/name,page_positions=65,object_bytes=1024,index_page_records=2,max_pages_per_object=3)
        archive_values.append(dict(directory=name,values=values.tolist()))
    example=Path(__file__).parent/'bridge_final_example'
    manifest=json.loads((example/'manifest.json').read_text())
    ids=rng.integers(0,manifest['positions'],10000,dtype=np.int64)
    reader=BatchReader(manifest,LocalRangeStore(example))
    (root/'chess_queries.json').write_text(json.dumps(dict(ids=ids.tolist(),values=reader.probe_many(ids).tolist()),separators=(',',':')))
    (root/'archives.json').write_text(json.dumps(archive_values,separators=(',',':')))
    from reference import counter_oracle
    graphs=[]
    for n in range(2,102):
        edges=[];seeds={}
        for i in range(n):
            neighbors=list(map(int,rng.integers(0,n,int(rng.integers(0,7)))))
            edges.append(neighbors)
            if not neighbors:seeds[i]=int(rng.choice([-1,0]))
        adjacency=np.zeros((n,n),bool)
        for i,neighbors in enumerate(edges):adjacency[i,neighbors]=True
        initial=np.zeros(n,np.int8)
        for i,value in seeds.items():initial[i]=value
        graphs.append(dict(edges=edges,seeds=[seeds.get(i) for i in range(n)],values=counter_oracle(adjacency,initial).tolist()))
    (root/'graph_inputs.json').write_text(json.dumps(graphs,separators=(',',':')))
    print(json.dumps(dict(frames=len(fixtures),archive_positions=sum(len(x['values']) for x in archive_values),chess_native_queries=len(ids))))

if __name__=='__main__':main()
