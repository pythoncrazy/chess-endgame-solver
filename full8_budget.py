"""Complete eight-piece budgets and scenarios for the current public R2 rate card.

Dataset count: Kirill Kryukov NULP, 2025-03-05. Storage estimate: tb8gen README.
Rates checked 2026-10-13 UTC. Units are decimal GB/TB/PB, not GiB/TiB/PiB.
This script does not predict symbolic compression or GPU runtime.
"""
import json
import math
from pathlib import Path


def costs(size, gets=0, puts=0, infrequent=False):
    gb=size/1e9
    if infrequent:
        return dict(storage_per_month=gb*.01,class_a=math.ceil(puts/1e6)*9,
                    class_b=math.ceil(gets/1e6)*.9,full_read_retrieval=gb*.01)
    return dict(storage_per_month=max(0,gb-10)*.015,
                class_a=math.ceil(max(0,puts-1e6)/1e6)*4.5,
                class_b=math.ceil(max(0,gets-10e6)/1e6)*.36,full_read_retrieval=0)


def model():
    n=38176306877748245;hour=3600;hbm=3.35e12;pcie=64e9
    rows=[]
    for size in (1.5e15,2e15,n*.25):
        rows.append(dict(bytes=size,pb=size/1e15,required_gb_per_s_one_hour=size/hour/1e9,
                         minimum_gpu_export_hours=size/pcie/3600,
                         standard=costs(size),infrequent=costs(size,infrequent=True),
                         one_gib_objects=math.ceil(size/(1<<30))))
    return dict(positions=n,required_states_per_s=n/hour,
                hbm_bytes_budget_per_state=hbm*hour/n,
                single_two_bit_read_write_hours=n*.5/hbm/3600,
                caveat='Read/write bound applies to a full conventional two-bit pass; compressed or symbolic algorithms need not perform that pass.',
                storage=rows)


if __name__=='__main__':
    data=model();Path('full8_budget.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
