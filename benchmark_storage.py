"""Measured exact codecs on native Syzygy symbol streams and chess DTM.

Native streams contain fillers for illegal slots and values that captures can
reconstruct. Thus they are NOT necessarily final WDL answers for legal boards.
Each sample has an explicit label. This script does not extrapolate the size of
all eight-piece tables.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from adaptive_storage import encode,decode,candidates,write_archive


def measure(values,page_positions=65536):
    baseline_bytes=adaptive_bytes=0;choices=Counter();n=len(values);start=time.perf_counter()
    for begin in range(0,n,page_positions):
        page=values[begin:begin+page_positions]
        options=candidates(page,level=19)
        baseline=min(len(f) for f,i in options if i['mode']=='offset-packed' and i['compression']=='zstd')
        frame,info=min(options,key=lambda c:len(c[0]));actual=decode(frame)
        assert np.array_equal(page,actual)
        baseline_bytes+=baseline;adaptive_bytes+=len(frame);choices[info['mode']+'/'+info['compression']]+=1
    return dict(positions=n,pages=(n+page_positions-1)//page_positions,page_positions=page_positions,
        baseline_frame_bytes=baseline_bytes,adaptive_frame_bytes=adaptive_bytes,
        reduction_percent=100*(1-adaptive_bytes/baseline_bytes) if baseline_bytes else 0,
        baseline='Offset bit-packed + Zstandard level 19, same frame header',
        choices=dict(choices),encode_all_candidates_seconds=time.perf_counter()-start,
        note='Frame bytes include 32-byte headers, exclude equal outer binary index and chess metadata.')


def native_streams(chess_path,table_path,cache_path,full_material='KQvKR'):
    sys.path.insert(0,str(Path(chess_path).resolve()))
    import chess.syzygy
    cache=Path(cache_path);cache.mkdir(parents=True,exist_ok=True);results=[]
    for name in ['KQvK','KRvK','KBNvK','KQvKR','KRvKB','KPvKP']:
        path=Path(table_path)/(name+'.rtbw');table=chess.syzygy.WdlTable(str(path));table.init_table_wdl()
        streams=[]
        if table.has_pawns:
            for f in range(4):
                for side in range(2):
                    if side==1 and not (table.data[4]&1):continue
                    streams.append((f,side,table.tb_size[2*f+side],table.files[f].precomp[side]))
        else:
            for side in range(2):
                if side==1 and not (table.data[4]&1):continue
                streams.append((0,side,table.tb_size[side],table.precomp[side]))
        labels=[];complete=name==full_material or name in ('KQvK','KRvK');total=sum(s[2] for s in streams)
        source_seconds=0
        for f,side,count,data in streams:
            selected=count if complete else min(count,8*8192)
            cachefile=cache/f'{name}-{f}-{side}-{selected}.npy'
            if cachefile.exists():values=np.load(cachefile)
            else:
                start=time.perf_counter()
                if not data.idxbits:values=np.full(selected,data.min_len,np.int32)
                else:
                    begins=[0] if complete else np.linspace(0,count-8192,8,dtype=np.int64)
                    spans=[range(count)] if complete else [range(int(b),int(b)+8192) for b in begins]
                    parts=[]
                    for span in spans:
                        parts.append(np.fromiter((table.decompress_pairs(data,i) for i in span),np.int32,count=len(span)))
                    values=np.concatenate(parts)
                source_seconds+=time.perf_counter()-start;np.save(cachefile,values)
            labels.append(values)
        values=np.concatenate(labels);result=measure(values)
        result.update(material=name,published_file_bytes=path.stat().st_size,total_stored_slots=total,
                complete_native_stream=complete,source_decode_seconds=source_seconds,
                native_symbol_values=list(map(int,np.unique(values))),streams=len(streams))
        if complete:
            result['comparison_warning']='Our data stream omits original chess headers/index semantics; include those before claiming a compatible replacement.'
        else:result['comparison_warning']='Selected contiguous blocks, not full material; original whole-file size is not a comparable sample denominator.'
        results.append(result);print(name,json.dumps(result),flush=True);table.close()
    return results


def reconstruction(chess_path,table_path):
    from validate_quotient import chess_check
    measured={}
    def experiment(nodes,graph,wdl,dtm):
        # Win proof depths are odd. Loss proof depths are even, with mate 0.
        win=np.flatnonzero(wdl==1);loss=np.flatnonzero(wdl==-1)
        assert np.all(dtm[win]%2==1) and np.all(dtm[loss]%2==0)
        stored={int(i):int(dtm[i]) for i in loss}
        rebuilt=np.full(len(wdl),-1,np.int64)  # Draw DTM is undefined, sentinel -1.
        for i in loss:rebuilt[i]=stored[int(i)]
        child_reads=0
        for i in win:
            defensive=[stored[j] for j in graph[i] if j in stored]
            assert defensive;rebuilt[i]=1+min(defensive);child_reads+=len(graph[i])
        assert np.array_equal(rebuilt,dtm)
        # No index claims: compare encoded value arrays in their current order.
        full=measure(dtm);loss_only=measure(dtm[loss]);half_loss=measure(dtm[loss]//2)
        measured.update(positions=len(nodes),draw_sink_included=True,winning_states=len(win),losing_states=len(loss),
            all_distances_reconstructed=True,full_dtm=full,loss_only_values=loss_only,
            parity_halved_loss_values=half_loss,quiet_child_inspections_to_reconstruct_wins=child_reads,
            caveat='DTM ignores fifty-move clock. Value-stream sizes exclude locating loss entries and WDL/index/dependency metadata; do not call their ratio complete-file savings.')
    validation=chess_check(chess_path,table_path,on_solved=experiment)
    measured['independent_wdl_positions']=validation['syzygy_positions_checked']
    return measured


def main():
    p=argparse.ArgumentParser();p.add_argument('--chess-path',required=True);p.add_argument('--table-path',required=True)
    p.add_argument('--cache-path',default='../compression_cache');p.add_argument('--output',default='storage_benchmark.json')
    p.add_argument('--skip-reconstruction',action='store_true');args=p.parse_args()
    result=dict(status='pass',h100_benchmarked=False,live_r2_tested=False,
        native_streams=native_streams(args.chess_path,args.table_path,args.cache_path))
    if not args.skip_reconstruction:result['exact_KQvK_DTM_reconstruction']=reconstruction(args.chess_path,args.table_path)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('wrote',args.output,flush=True)


if __name__=='__main__':main()
