"""R2/Workers read-only rate model, checked against official example math.

Rates verified 2026-10-13. Hit rates, traffic and CPU times are assumptions,
not demonstrated deployment measurements. No inference about cache hit rate
from dataset compression or chess knowledge is justified.
"""
import json
import math
from pathlib import Path


def monthly(queries,storage_bytes=0,blocks_per_position=1,block_hit_rate=0,
            positions_per_payload_get=1,metadata_gets_per_position=0,
            fetched_bytes_per_get=65536,storage_class='standard',
            positions_per_api_request=1,worker_cpu_ms_per_api_request=5):
    if storage_class not in ('standard','infrequent'):raise ValueError('unknown storage class')
    if not 0<=block_hit_rate<=1 or positions_per_payload_get<=0 or positions_per_api_request<=0:
        raise ValueError('invalid traffic assumptions')
    if min(queries,storage_bytes,blocks_per_position,metadata_gets_per_position,
           fetched_bytes_per_get,worker_cpu_ms_per_api_request)<0:raise ValueError('negative assumption')
    payload_gets=queries*blocks_per_position*(1-block_hit_rate)/positions_per_payload_get
    total_gets=payload_gets+queries*metadata_gets_per_position
    gb=storage_bytes/1e9;standard=storage_class=='standard'
    storage=max(0,math.ceil(gb)-(10 if standard else 0))*(.015 if standard else .01)
    read_operations=math.ceil(max(0,total_gets-(10_000_000 if standard else 0))/1e6)*(.36 if standard else .9)
    fetched=total_gets*fetched_bytes_per_get
    retrieval=0 if standard else math.ceil(fetched/1e9)*.01
    api_requests=math.ceil(queries/positions_per_api_request)
    worker_requests=max(0,api_requests-10_000_000)/1e6*.30
    worker_cpu=max(0,api_requests*worker_cpu_ms_per_api_request-30_000_000)/1e6*.02
    worker=5+worker_requests+worker_cpu
    return dict(queries=queries,expected_payload_gets=payload_gets,expected_all_r2_gets=total_gets,
       expected_fetched_bytes=fetched,api_requests=api_requests,r2_storage_usd=storage,
       r2_read_operations_usd=read_operations,r2_retrieval_usd=retrieval,worker_usd=worker,
       estimated_monthly_total_usd=storage+read_operations+retrieval+worker,
       assumptions=dict(blocks_per_position=blocks_per_position,block_hit_rate=block_hit_rate,
          positions_per_payload_get=positions_per_payload_get,metadata_gets_per_position=metadata_gets_per_position,
          fetched_bytes_per_get=fetched_bytes_per_get,storage_class=storage_class,
          positions_per_api_request=positions_per_api_request,worker_cpu_ms_per_api_request=worker_cpu_ms_per_api_request))


def report():
    # Official R2 1,000 GB example after the Standard free storage allowance.
    assert monthly(0,storage_bytes=1000*10**9)['r2_storage_usd']==14.85
    # Official Workers 100M request, 7ms per request example is $45.40.
    assert abs(monthly(100_000_000,block_hit_rate=1,worker_cpu_ms_per_api_request=7)['worker_usd']-45.4)<1e-9
    samples={}
    for hit in (0,.9,.99):
        samples[f'100M_single_block_hit_{hit}']=monthly(100_000_000,block_hit_rate=hit)
    samples['100M_four_blocks_hit_99pct']=monthly(100_000_000,blocks_per_position=4,block_hit_rate=.99)
    samples['100M_batched_32_positions']=monthly(100_000_000,block_hit_rate=.99,
        positions_per_api_request=32,worker_cpu_ms_per_api_request=100)
    marginal=[dict(cache_hit_rate=h,r2_gets_per_million_positions=1e6*(1-h),
                   r2_standard_read_usd_before_free_tier=.36*(1-h)) for h in (0,.9,.99,.999)]
    return dict(as_of_utc='2026-10-13',status='pass',examples=samples,marginal_one_required_block=marginal,
        ia_storage_vs_standard_retrieval_break_even_fraction=.5,
        warning='Examples exclude data storage unless supplied, writes, logging, other infrastructure and deployment work. Metadata assumed warm and free allowances available. A full chess probe can need several dependent blocks. IA input must reflect billable retrieval bytes, not merely logical result size. R2 unit rounding is modelled; Workers uses stated unit rates. No latency or hit-rate promise.',
        sources=['https://developers.cloudflare.com/r2/pricing/','https://developers.cloudflare.com/workers/platform/pricing/'])


if __name__=='__main__':
    result=report();Path('lookup_costs.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
