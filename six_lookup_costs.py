"""Origin-only cost sensitivity from measured traces of native probes."""
import json
from pathlib import Path

def model():
    root=Path(__file__).parent
    measured=json.loads((root/'native_six_validation.json').read_text())
    rows=[]
    for profile in measured['profiles']:
        for phase in profile['results']:
            gets=phase['origin_reads']/phase['positions']
            rows.append(dict(cache_budget_bytes=profile['cache_budget_bytes'],mode=profile['mode'],phase=phase['phase'],
                observed_origin_reads=phase['origin_reads'],observed_positions=phase['positions'],
                observed_bytes=phase['origin_bytes'],origin_gets_per_position_in_this_trace=gets,
                marginal_standard_origin_dollars_per_million_positions_if_trace_rate_repeated=gets*.36,
                marginal_IA_origin_dollars_per_million_positions_if_trace_rate_repeated=gets*.90+phase['origin_bytes']/phase['positions']*1e6/1e9*.01))
    big=[p for p in measured['profiles'] if p['cache_budget_bytes']==64<<20]
    scalar=next(p['results'][0] for p in big if p['mode']=='scalar')
    batch=next(p['results'][0] for p in big if p['mode']=='batch')
    saved=scalar['origin_reads']-batch['origin_reads']
    extra=batch['seconds']-scalar['seconds']
    result=dict(trace_rows=rows,
        cold_batch_read_reduction_fraction=saved/scalar['origin_reads'],
        illustrative_extra_serial_origin_latency_break_even_ms=max(0,extra)/saved*1000 if saved>0 else None,
        caveats=['These are this small corpus/cache trace, not collection-wide hit rates.',
                 'Repeating an identical cold trace with a retained cache does not repeat its cold miss rate.',
                 'Price arithmetic excludes monthly free allowances, billing rounding, API/Worker/VM compute, transport and catalog transfer.',
                 'Latency break-even adds serial request delay to local trace time; it does not model parallel requests, dependency waves or actual cloud latency.',
                 'IA storage savings need a separate monthly retrieval/operation access model.'],
        rate_source='https://developers.cloudflare.com/r2/pricing/',live_r2_measured=False)
    (root/'six_lookup_costs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__': model()
