"""H100 preflight and exact operator comparison. Never invents generation times."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path

def run(output,fibers,toolkit):
    root=Path(__file__).parent.resolve();result=dict(status='blocked',whole_six_piece_solver=False,h100_generation_measured=False)
    smi=shutil.which('nvidia-smi')
    if not smi:result['reason']='No NVIDIA driver/GPU is exposed: nvidia-smi is unavailable.'
    else:
        result['gpu_inventory']=subprocess.check_output([smi,'--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
        if 'H100' not in result['gpu_inventory']:result['reason']='This benchmark profile requires an H100; inventory does not identify one.'
        elif not (Path(toolkit)/'include'/'cuda.h').exists():result['reason']='CUDA toolkit headers are absent at the supplied path; Hopper cuTile needs Tile IR 13.3 or newer.'
        else:
            cargo=shutil.which('cargo')
            if not cargo:result['reason']='Cargo is unavailable.'
            else:
                env=os.environ.copy();env['CUDA_TOOLKIT_PATH']=toolkit
                command=[cargo,'run','--locked','--release','--manifest-path',str(root/'rust'/'gpu'/'Cargo.toml'),'--bin','fiber_compare','--',str(fibers)]
                start=time.perf_counter();completed=subprocess.run(command,env=env,text=True,capture_output=True)
                result['build_and_run_wall_seconds']=time.perf_counter()-start
                if completed.returncode:result['reason']='GPU build/JIT/run failed; no performance result accepted.';result['stderr_tail']=completed.stderr[-6000:]
                else:
                    result['operator_measurements']=json.loads(completed.stdout.strip().splitlines()[-1]);result['status']='operator-benchmark-complete'
                    result['reason']='Resident operators only: excludes full move-mask generation, indexing, global fixpoint, distances and durable output.'
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(Path(__file__).parent/'h100_run_status.json'));p.add_argument('--fibers',type=int,default=1<<22);p.add_argument('--toolkit',default='/usr/local/cuda-13.3');a=p.parse_args();run(a.output,a.fibers,a.toolkit)
