The first complete GPU generation benchmark is **KQBNvKR**, a six-piece pawnless
material. It contains all legal placements without castling rights, for both
turns. This is the same domain as the published Syzygy comparison. KNNvKNN is a
second benchmark, with repeated pieces and a good fit for tensor operations.
KRPvKRP is next, for pawn double moves, promotions and en passant.

One material and a complete six-piece collection are different jobs. We tested
the operators and the probes. We did **not** do a complete six-piece GPU
generation run.

The first generation metric is unclocked WDL, with capture dependencies that are
solved independently. The sign of the published five-class Syzygy values can
verify that metric. The next stages must separately implement and verify DTM or
Syzygy-compatible WDL50/DTZ50. Unclocked WDL does not give those distances or
draw-rule values. Neither a fresh-clock table nor a FEN gives arbitrary
repetition history.

Castling-right states must have a separate exact extension. This extension must
have legal castling transitions and a symmetry that keeps the castling rights.
The first comparison with published data does not include those states. Do not
say that it solves them.

The measured reference facts are in `six_measurements.json`. The KQBNvKR native
file has **18,577,168 bytes** and **13,531,664,160 native index slots**. KNNvKNN
uses **1,013,968 bytes** and **1,546,346,340 stored slots**. Color symmetry and
repeated pieces make its native storage domain smaller.

These numbers are native streams, with invalid fillers and codes that the reader
reconstructs from captures. They are not a count of legal boards.
Uniform/stratified compression samples have explicit labels. We do not
extrapolate their sizes to a complete custom archive.

The hardware model uses H100 **SXM**, 80 GB and 3.35 TB/s. Dense FP16 is 989.5
TFLOPS. Dense INT8 is 1,979 TOPS. The advertised figures with the sparsity
footnote are 1,979 FP16 / 3,958 INT8. Do not use them as dense rates. This model
does not describe an H100 PCIe card, which has a different bandwidth/performance.

Rust and cuTile are the implementation target. Absent CUDA dependencies in the
development environment now block the GPU source build.

For the GPU working layout, keep canonical nonadjacent king pairs. Pad the four
other piece coordinates to 64 values each. Both turns give
`2 * 462 * 64^4 = 15,502,147,584` candidate cells. This is about 14.6% more than
the published native slot count.

Do not remove duplicate diagonal-stabilizer orientations from quiet fibers. This
redundancy does not cause errors, and it keeps the local move graph regular.
King moves normalize the king pair and transform all piece coordinates. The
output uses the compact canonical native/storage index. The legality mask
excludes collisions and invalid check states.

This layout index now has a CPU Rust ranker and an independent closure fixture
generator. The checks include all 4,096 king-pair inputs and 4,566 legal quiet
transitions, with 946 king moves. They also include 2,053 random/boundary ID
roundtrips.

A transition of a piece other than a king changes one base-64 digit and the turn
directly. King transitions transform the four other coordinates through small
lookup tables. This verifies the padded quiet index. It does not verify capture
dependencies or an end-to-end GPU solve. Native storage uses its own maintained
compact ranker. The padded index is not a Syzygy file index.

The bounded WDL working model uses about **28.3 GB**. It contains current/next
two-bit state, validity, and static legal-move and capture-boundary masks. It
also contains one bit for each dynamic move family, which shows a child that is
not a win. It also has a lower-table allowance and tile scratch.

A full 16-bit all-side DTM buffer adds about 31.0 GB. A single-side 16-bit DTZ
buffer adds about 15.5 GB. Distances that are larger than the narrow
representation must have exact overflow escapes or a wider/streamed stage. Never
saturate them silently.

The exact quiet operator fixes five pieces and puts the sixth on each of 64
squares. For a mover other than a king, physical source validity and destination
own-king safety are separate geometric factors. The operator excludes captures.
The boundary path processes them.

The CPU fixture suite independently checks the full legal relation for knight,
bishop, rook and queen fibers with both colors. These checks include positions
in check. They compare each of the 64×64 adjacency cells in each of 320 fibers.

For knights, `A` is the **same 64×64 matrix** for all board skeletons. Each column
is a different real fiber with its own masks and child state. Compute

`B[j,g] = safe[j,g] * (1 + 9*[child != Win] + 81*[child == Loss])`.

The INT8 values are 0, 1, 10 or 91. Each knight row has a maximum of eight
destinations, so the base-9 digits of `Z=A*B` are exact child counts. The three
digits count the legal children, the children that are not wins and the children
that are losses. The maximum is 728.

INT32 accumulation is exact. FP16 inputs with FP32 accumulation also represent
these values/sums exactly. This gives a comparison kernel. FP8 E4M3 **cannot**
represent 91 exactly. Do not use it here.

Decode the three digits. Apply source validity. Pack the output masks. This is
the reference scheme.

The faster iterative scheme uses **0/16/17** instead:
`B = safe * (16*[child != Win] + [child == Loss])`. Thus `Z & 15` detects a child
that is a loss, and `Z != 0` detects a child that is not a win. The maximum sum
is 136. This scheme removes division/remainder and writes only two dynamic
bitplanes. FP8 E4M3 cannot represent 17 exactly.

Cache move existence separately. A zero output alone must never classify
stalemate as loss. The design does not store a large state array of float values
or a per-fiber adjacency in HBM.

Another variant uses **0/8/9**, which FP8 E4M3 can represent exactly:
`B = safe * (8*[child != Win] + [child == Loss])`. Then `Z & 7` detects one to
seven destinations that are losses. Eight losses carry into the next digit, but
only eight losses give `Z == 72`. This is because eight is the maximum degree.
Therefore `win = ((Z & 7) != 0 || Z == 72)` and `nonwinning = (Z != 0)` are exact.

The CPU Rust suite enumerates all 65,536 child-state assignments at a
degree-eight source. Each child is illegal, a win, unresolved or a loss. The
enumeration includes this carry and the cases with no legal move. The GPU comparison source
uses this common scheme for INT8/INT32, FP16/FP32 and FP8/FP32 candidates. It
requests ordinary accumulation, without the optional reduced-precision fast FP8
mode.

Input conversion and bounded FP32 integer sums are exact in the CPU model. We did
not test the compiled device arithmetic. The listed FP8 peak is an optimistic
ceiling. It is not the achieved throughput of strict accumulation.

The knight matrix also has a maximum of two nonzeros in each contiguous
four-column group. Thus a 2:4 sparse representation can keep it exactly, with
zeros in the unused slots. The current cuTile API/source benchmark does **not**
implement that sparse path. Treat it as a cuda-oxide/PTX experiment, not as an
achieved 2× speedup. Its value depends on the actual pack/reduction/register
overhead.

The alternative knight operator is eight masked bit shifts. Rook/bishop/queen
operators use occluded fills with 1/2/4-square stages. These fills stop at
**physical blockers**. Destination safety masks filter the endpoints. An unsafe
intermediate square does not block a ray. This method makes all 64 source
predicates without a 64×64 legal-move matrix.

These integer stencils have a lower advertised arithmetic throughput than tensor
cores. But they are serious competitors to tensor cores.

Keep the knight coordinate as the contiguous bitboard axis. Other piece axes must
have a 64×64 bit transpose in a tile. If possible, fuse those transposes with
loads and packed candidate output. If you materialize the layouts instead, charge
the full read/write traffic of each transpose to the roofline. Pack/scatter so
that each CTA writes only the output word that it owns. Do not hide uncoalesced
or atomic traffic in an ideal one-byte-per-row calculation.

Use monotone WDL values and idempotent dirty tile notifications. Each family
caches if a parent has a child that is not a win. A stale cached `true` delays a
loss. A cached `false` stays sound, because a proven child win never reverts.
Family contributions that are not yet computed stay conservative.

A parent wins when a legal child loses. It loses only when it has a legal move
and the `nonwinning` mask of each family is false. Checkmate and stalemate are
different seeds. Unresolved cycles are draws only after complete global
quiescence. This quiescence includes delivered boundaries and notifications.
Duplicate messages must not decrement counters two times.

Dense dirty tiles can batch independent knight columns for MMA. Sparse late
frontiers and king/capture rank calculations use SIMT. Select the crossover from
measured whole-tile times, not from TFLOPS.

For larger jobs, shard owners save exact boundary states and epochs. Termination
must include all in-flight work. Do not do an R2 GET in the iterative GPU fixed
point. Decode the necessary lower tables into resident device caches before the
timed solve.

Those device dependencies must contain final, correctly probed WDL values for
legal boards. An exact compatible GPU capture prober is the other correct option.
Raw native stored codes have fillers and capture reconstruction rules. Do not
copy their signs directly into generation seeds. Decode/prepare by blocks, with
constant and grammar-run expansion. Do not call a scalar CPU prober for each slot.

Precompute the king-move transforms. Each canonical king pair, turn and adjacent
king square maps to a new pair and one of eight board transforms. The compact
table and the 8×64 square-transform LUT fit in cache. Apply the transform, then
rank the four other coordinates. You must still check the actual attack/capture
legality.

Asynchronous WDL commitment depth is only a proof rank. It is not necessarily the
optimal DTM. Compute exact distances in an ordered/layered or separately
validated stage.

The resident packed INT8 and packed bitboard knight operators have an optimistic
footprint of **0.75 bytes per source row**. This value is for the
two-output-word benchmark. Thus HBM limits each of them to about **4.47 trillion
source rows/s**. The dense arithmetic limit of packed INT8 is about 15.5 trillion
rows/s. These two limits do not include legality-mask construction, transpose,
scalar pack instructions, spills, king moves, capture ranks or global
convergence. A source row for one piece is **not** an evaluation of a whole
position.

An example INT32 issue budget is 16.75 trillion lane instructions/s. This budget
uses the 64 INT32 lanes/SM of the H100, compared with 128 FP32 lanes/SM. It also
uses the convention of two operations/FMA in the 67 TFLOPS FP32 peak. At
8/16/32/64 scalar lane instructions per source row, this gives
2.09/1.05/0.52/0.26 trillion rows/s. These rates are before occupancy and other
limits. These are sensitivity cases, not measured kernel instruction counts.

Pack operations through warp ballots can be faster than weighted 64-bit tile
reductions. Inspect the emitted code. Measure it. A bitboard stencil does many
operations on 64 source rows at once. Sparse MMA or more tensor TOPS cannot
remove the cost of scalar pack operations.

For a runtime plan, use `T = N * equivalent_sweeps / full_position_rate`. Then
add initialization, lower-table preparation, rule/distance stages, checks,
compression and durable output.

This example uses 100 equivalent sweeps over the padded layout. Then 1/5/20
billion **complete** position evaluations per second take about 25.8/5.2/1.3
minutes of compute. At 20 sweeps, those values are five times smaller. These are
sensitivity scenarios, not measured or assured throughputs. The necessary round
count is unknown until the complete solver runs.

Instrument each phase with CUDA events and host wall time. Do a preflight check
of the actual GPU variant, toolkit, driver, memory, clocks and power
configuration. Do a JIT warm-up before the operator time measurements. Report the
JIT time separately. Use a working set that is much larger than L2. Record the
useful source rows, real legal edges, dirty-tile density, physical HBM bytes,
spills and boundary traffic.

Measure INT8, FP16, packed stencils and their layout costs on identical inputs.
FMA/TOPS and nominal logical bytes are not Nsight counters. The cuTile benchmark
validates all results against the CPU oracle before it measures time. But we did
not execute it on an H100 here.

For durable storage, keep **native Syzygy compression as the baseline**. At
16,384 positions/page, our old 32-byte fixed records alone are 26.4 MB for
KQBNvKR. This is already larger than its full 18.6 MB native file. Do not pay for
a per-logical-page header on very compressible materials.

The new native range format adds one SHA-256 hash per **physical 16 KiB** tile
(about 0.195%). The catalog authenticates the hash pages. The compiled Rust
filesystem implements exact random reads through shared caches and the
maintained native Syzygy rank/probe library. It keeps the capture semantics and
the published five-class rules.

Use physical hash/range pages and warmed header/grammar/index caches. Keep the
source objects content-addressed. Reject corrupt data and corrupt hash pages
before you decode them. The wrapper keeps the immutable `.rtbw/.rtbz` bytes and
does not transcode them.

A new GPU-generated result can first use exact packed/Zstd stream blocks. Then it
can compete with Syzygy-style compression on complete files. The final
representation selector must count the payload. It must also count authenticated
indexes, dictionaries, route and rule metadata. A larger compute-friendly working
tensor does not mean a larger final object collection.

The measured current six-piece mirror lists 365 WDL files with a total of
**72.83 GB**. It also lists 365 DTZ files with a total of **87.39 GB** (decimal).
Together, this is about 160.23 GB of native data. Physical hash indexes add about
0.2%, plus the catalogs.

At the listed R2 Standard price of $0.015/GB-month, this is approximately
**$2.41/month before the free storage allowance and bill round-off**. This value
does not include requests/API compute. This value is for the present six-piece
collection. It is not an eight-piece estimate.

A cold read can still fetch several metadata/data tiles and capture dependencies.
Do not extrapolate the warm-cache rates from the small test to a full collection.

The lower three/four/five-piece directories add 396,105,616 WDL bytes and
587,852,304 DTZ bytes. The listed full collection through six pieces has
**161,209,573,952 native bytes**. This is about **161.52 GB with physical
hashes**, plus small catalogs. For a constant stored quantity for the whole
month, this rounds to 162 GB-month. This gives **$2.43/month before free
storage**, or **$2.28 if the full 10 GB allowance is available**. This value does
not include operations and API compute.

`native_six_validation.json` records 1,005 independently checked FENs and native
WDL/DTZ fixtures. It records cold/warm passes at 64 MiB and 1 MiB cache budgets.
It also records eight concurrent batch sessions, evictions, absent dependencies
and corrupt/short HTTP resources.

In the 64 MiB trace, batch mode decreases cold origin reads **565 -> 448**. The
fetched bytes are the same in both modes, 8,927,936 bytes. Both modes have zero
warm origin reads. Batch mode adds local CPU work. Thus scalar mode is still
better for a cheap local store.

The marginal R2 Standard origin cost is about **$0.202 -> $0.160 per million
positions**. This cost is correct only *if those cold trace miss rates occur at
scale*. It is before free operations and bill round-off. These are
trace-dependent origin costs, not live cloud service prices. You must still
measure the actual request/compute cost and cache locality. Refer to
`six_lookup_costs.json`.
