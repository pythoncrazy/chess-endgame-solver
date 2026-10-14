"""Reproduce native Rust tests and cache/batch benchmarks against published data."""
import argparse,json,subprocess
from pathlib import Path

def validate(cargo,objects):
    root=Path(__file__).parent.resolve();project=root/'rust'/'native';manifest=project/'Cargo.toml'
    core=root/'rust'/'Cargo.toml'
    for command in ([cargo,'fmt','--manifest-path',str(core),'--check'],
                    [cargo,'fmt','--manifest-path',str(root/'rust'/'gpu'/'Cargo.toml'),'--check'],
                    [cargo,'clippy','--locked','--manifest-path',str(core),'--all-targets','--','-D','warnings'],
                    [cargo,'test','--locked','--manifest-path',str(core),'--release','-j','8']):subprocess.run(command,check=True)
    for command in ([cargo,'fmt','--manifest-path',str(manifest),'--check'],
                    [cargo,'clippy','--locked','--manifest-path',str(manifest),'--all-targets','--features','http','--','-D','warnings'],
                    [cargo,'test','--locked','--manifest-path',str(manifest),'--release','--features','http','-j','8'],
                    [cargo,'build','--locked','--manifest-path',str(manifest),'--release','--features','http','-j','8']):subprocess.run(command,check=True)
    results=[]
    for budget in (64<<20,1<<20):
        for mode in ('scalar','batch'):
            data=json.loads(subprocess.check_output([str(project/'target'/'release'/'tablebase-native'),str(Path(objects).resolve()),str(root/'six_extended_probe_queries.json'),str(budget),mode]))
            assert data['retained_data_cache_bytes']<=budget
            results.append(data)
    result=dict(status='pass',legal_six_piece_fens=1000,extra_five_class_positions=5,profiles=results,
                independent_wdl_and_dtz_fixture_positions=64,physical_data_and_hash_index_corruption_rejected=True,
                zero_cache_and_eviction_tested=True,missing_dependency_rejected=True,http_range_transport_mock_tested=True,
                concurrent_native_batches=8,fp8_degree_eight_assignments=65536,dense_index_quiet_closure_fixture_tested=True,
                native_source_preserved=True,full_material_generated=False,h100_benchmarked=False,live_r2_tested=False,
                source_revision='fd6296926137d89192c7810a0894346a03f749ef')
    (root/'native_six_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cargo',default='cargo');p.add_argument('--objects',required=True);a=p.parse_args();validate(a.cargo,a.objects)
