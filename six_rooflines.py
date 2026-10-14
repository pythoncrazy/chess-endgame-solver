"""Six-piece H100 end-to-end bounds, with explicit traffic and timing assumptions."""
import json,math,re,hashlib
from pathlib import Path

def model():
    root=Path(__file__).parent;measured=json.loads((root/'six_measurements.json').read_text())
    native=next(x for x in measured['materials'] if x['material']=='KQBNvKR')['native_slots']
    dense=2*462*64**4
    hbm=3.35e12;fp16=1979e12/2;int8=3958e12/2;pcie=64e9
    kernels=[]
    for name,traffic,ops,peak in [
        ('naive material-specific FP16 relation + expanded output',128+2+4,128,fp16),
        ('shared knight FP16 with expanded RHS and int32 output',2+4,128,fp16),
        ('shared knight INT8 with expanded RHS and int32 output',1+4,128,int8),
        ('fused packed-input INT8 knight, two packed output words',.75,128,int8),
        ('fused packed-input FP16 knight, two packed output words',.75,128,fp16),
        ('fused packed-input FP8 knight, two packed output words',.75,128,int8),
        ('packed knight stencil, four input/two output words',.75,0,0),
    ]:
        compute=peak/ops if ops else None;bandwidth=hbm/traffic
        rate=min(compute,bandwidth) if compute else bandwidth
        kernels.append(dict(name=name,nominal_hbm_bytes_per_source_row=traffic,dense_equivalent_ops_per_row=ops,
                            peak_compute_rows_s=compute,peak_bandwidth_rows_s=bandwidth,
                            conditional_rate_ceiling_rows_s=rate,full_dense_source_pass_lower_bound_ms=1000*dense/rate,
                            caveat='Resident operator only; mask generation, layout changes, rank/gathers, packing instructions, spill traffic, global convergence and other pieces are excluded. Static shared matrix assumed hot in cache. FP8 peak is optimistic; full-precision accumulation can reduce attainable arithmetic throughput.'))
    # An illustrative INT32 instruction-issue cap. Hopper has half as many
    # INT32 lanes as FP32 lanes; advertised FP32 FLOPS count two ops per FMA.
    # Actual compiled instruction counts, dual issue and 64-bit lowering must
    # be measured, so these are sensitivity cases, not kernel predictions.
    scalar32_issue=67e12/4
    scalar_sensitivity=[dict(assumed_int32_lane_instructions_per_source_row=n,
                             conditional_scalar_rows_s=scalar32_issue/n,
                             ceiling_with_packed_memory_rows_s=min(scalar32_issue/n,hbm/.75))
                        for n in (8,16,32,64)]
    working={
        'current_and_next_WDL_two_bit_each':dense*.5,
        'validity_one_bit':dense/8,
        'has_legal_move_one_bit':dense/8,
        'five_dynamic_nonwinning_family_masks':dense*5/8,
        'static_capture_nonwinning_one_bit':dense/8,
        'lower_material_WDL_allowance':1e9,
        'tile_queues_and_metadata_allowance':64e6,
        'temporary_tiles_and_output_allowance':4e9,
    }
    memory=sum(working.values())
    scenarios=[]
    for equivalent_sweeps in (20,100,300):
        for rate in (1e9,5e9,20e9):
            scenarios.append(dict(equivalent_full_position_evaluations=equivalent_sweeps,
                assumed_complete_position_evaluations_s=rate,
                dense_working_compute_seconds=dense*equivalent_sweeps/rate,
                native_compact_compute_seconds=native*equivalent_sweeps/rate,
                assumption='Whole-position rate includes all legal successors and indexing, not one piece/operator. Scenarios are not empirical predictions. Add initialization, lower-table preparation, distance/rule stages, validation and compression/output.'))
    rounds=[]
    for bytes_per_position in (8,64,1024):
        for efficiency in (.25,.5,1):
            rounds.append(dict(aggregate_bytes_per_position_per_round=bytes_per_position,
                               assumed_hbm_efficiency=efficiency,
                               one_dense_round_seconds=dense*bytes_per_position/(hbm*efficiency),
                               hundred_dense_rounds_seconds=100*dense*bytes_per_position/(hbm*efficiency)))
    directories=[]
    for tier,prefix,metric,suffix in [('6','six','wdl','rtbw'),('6','six','dtz','rtbz'),
                                     ('3-4-5','lower','wdl','rtbw'),('3-4-5','lower','dtz','rtbz')]:
        path=root/'reference_directory_listings'/f'{tier}-{metric}.html'
        if path.exists():
            rows=re.findall(r'href="([^"/]+\.'+suffix+r')"[^\n]*?([0-9]+)\s*\n',path.read_text())
            size=sum(int(n) for _,n in rows)
            directories.append(dict(metric=metric,piece_tier=tier,files=len(rows),bytes=size,decimal_gb=size/1e9,gib=size/2**30,
                                    physical_sha256_index_bytes=sum(32*math.ceil(int(n)/16384) for _,n in rows),
                                    listed_standard_storage_monthly_before_free_allowance=size/1e9*.015,
                                    source=f'https://tablebase.lichess.ovh/tables/standard/{tier}-{metric}/',listing_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    native_total=sum(d['bytes'] for d in directories)
    indexed_total=native_total+sum(d['physical_sha256_index_bytes'] for d in directories)
    return dict(hardware=dict(gpu='H100 SXM 80GB',memory_bytes=80e9,hbm_bytes_s=hbm,dense_fp16_ops_s=fp16,
                              dense_int8_ops_s=int8,dense_fp8_ops_s=int8,assumed_pcie_one_direction_bytes_s=pcie,
                              source='https://www.nvidia.com/en-us/data-center/h100/',note='Public TC peaks marked with sparsity were divided by two. PCIe is a nominal one-direction ceiling, not measured application throughput.'),
                material='KQBNvKR',native_index_slots=native,padded_king_pair_dense_slots=dense,
                dense_slot_expansion_ratio=dense/native,working_memory_bytes=working,
                estimated_WDL_working_bytes=memory,additional_u16_DTM_all_sides_bytes=2*dense,
                additional_u16_DTZ_one_side_bytes=dense,estimated_WDL_plus_DTM_working_bytes=memory+2*dense,
                distance_caveat='16-bit buffers require explicit overflow/undefined encodings; they are not a universal bound on distance. DTM and DTZ50 need distinct correct algorithms.',
                resident_operator_rooflines=kernels,aggregate_round_memory_sensitivity=rounds,
                tensor_packing_scalar_sensitivity=dict(illustrative_int32_lane_instructions_s=scalar32_issue,cases=scalar_sensitivity,
                    caveat='Not measured SASS. Unpack/select/reduce/repack, 64-bit lowering and register occupancy can dominate. Compare warp ballot packing with tile reductions on hardware. A packed stencil amortizes many operations over 64 source rows.'),
                whole_position_throughput_scenarios=scenarios,
                generation_export_bounds=dict(two_bit_dense_WDL_export_at_nominal_PCIe_seconds=dense*.25/pcie,
                                               conditional_full_raw_cycle_export_read_write_PCIe_seconds=dense*.5/pcie,
                                               note='Keep the fixed point on the GPU. Do not transfer the full state every round. Final compression can overlap output, but its throughput must be measured.'),
                existing_six_piece_collection=directories,
                through_six_piece_collection=dict(native_bytes=native_total,physical_hash_index_bytes=indexed_total-native_total,
                    bytes_before_small_catalogs=indexed_total,rounded_standard_monthly_before_free_allowance=math.ceil(indexed_total/1e9)*.015,
                    rounded_standard_monthly_if_full_10GB_allowance_available=max(0,math.ceil(indexed_total/1e9)-10)*.015,
                    caveat='Listed directory sizes, not downloaded/generated whole collection. Whole-month constant storage assumed; catalogs, writes, requests and API compute additional. Prices depend on remaining account allowances.'),
                h100_benchmarked=False,complete_gpu_six_piece_solver_implemented=False,
                conclusion='Packed local stencils and packed-input INT8 tensor kernels share a memory roof. The latter only accelerates eligible operators. No complete end-to-end runtime follows from TOPS alone.')

if __name__=='__main__':
    result=model();path=Path(__file__).parent/'six_rooflines.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(material=result['material'],native_slots=result['native_index_slots'],dense_slots=result['padded_king_pair_dense_slots'],wdl_working_gb=result['estimated_WDL_working_bytes']/1e9,wdl_plus_dtm_working_gb=result['estimated_WDL_plus_DTM_working_bytes']/1e9,operator_roofs=result['resident_operator_rooflines']),indent=2))
