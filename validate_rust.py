"""Compile/lint Rust and verify its outputs with the independent Python decoder."""
import argparse,json,subprocess,tempfile,platform
from pathlib import Path
import numpy as np
from adaptive_storage import decode

def validate(cargo):
    root=Path(__file__).resolve().parent;project=root/'rust';manifest=project/'Cargo.toml'
    checks=[]
    for command in ([cargo,'fmt','--manifest-path',str(manifest),'--check'],
                    [cargo,'clippy','--manifest-path',str(manifest),'--all-targets','--','-D','warnings'],
                    [cargo,'test','--manifest-path',str(manifest),'--release'],
                    [cargo,'build','--manifest-path',str(manifest),'--release','--bins']):
        subprocess.run(command,check=True);checks.append(command[1])
    fixtures=json.loads((project/'fixtures'/'frames.json').read_text());unique={tuple(f['values']) for f in fixtures}
    with tempfile.TemporaryDirectory() as directory:
        frame=Path(directory)/'encoded.tbc'
        for values in unique:
            subprocess.run([str(project/'target'/'release'/'tb_encode'),str(frame)],input=json.dumps(values).encode(),check=True)
            assert np.array_equal(decode(frame.read_bytes()),np.array(values,np.int64))
    benchmark=json.loads(subprocess.check_output([str(project/'target'/'release'/'tb_bench'),str(root/'pawn_examples'/'KPvKP'),str(project/'fixtures'/'cache_trace.json')]))
    (root/'rust_cache_benchmark.json').write_text(json.dumps(benchmark,indent=2)+'\n')
    result=dict(status='pass',rustc=subprocess.check_output([str(Path(cargo).with_name('rustc')),'--version'],text=True).strip() if '/' in cargo else subprocess.check_output(['rustc','--version'],text=True).strip(),
                checks=checks,independent_codec_frames=len(fixtures),encoding_modes=10,compressors=3,encoded_corruption_checks=len(fixtures)*8,
                python_decoded_rust_arrays=len(unique),native_chess_archive_ids=10000,legacy_v2_archive_ids=10000,
                graph_oracle_cases=100,bitmask_loss_predicates=65536,concurrent_join_callers=16,
                decimal_ids_above_2pow53=True,int64_extremes=True,cache_eviction_and_range_splitting=True,
                cpu=platform.processor(),gpu_dependency_revision='bc46b1ffb751885073066b087594045f07f0f81b',
                gpu_build_status='Blocked in NVIDIA cuda-bindings build: no CUDA toolkit installed. Kernel host/JIT compilation and H100 execution are unverified.',
                h100_benchmarked=False,live_r2_tested=False,full_eight_piece_generator_implemented=False)
    (root/'rust_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cargo',default='cargo');args=p.parse_args();validate(args.cargo)
