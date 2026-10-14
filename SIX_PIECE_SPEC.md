The first complete GPU generation benchmark is **KQBNvKR**, a six-piece pawnless
material, including all legal placements without castling rights and both turns,
matching the published Syzygy comparison domain. KNNvKNN is a second
tensor-friendly, repeated-piece benchmark. KRPvKRP follows for pawn double moves,
promotions and en passant. One material and a complete six-piece collection are
different jobs. We have tested operators and probing; a complete six-piece GPU
generation run has **not** been performed.

The initial generation metric is unclocked WDL, with independently solved capture
dependencies. Published five-class Syzygy values can verify that metric by their
sign. The next stages must separately implement/verify DTM or Syzygy-compatible
WDL50/DTZ50. Unclocked WDL does not determine those distances or draw-rule values.
Neither a fresh-clock table nor a FEN determines arbitrary repetition history.
Castling-right states require a separate exact extension, with rights-preserving
symmetry and legal castling transitions. The first published-data comparison
does not cover those states; it must not be advertised as solving them.

Measured reference facts are in `six_measurements.json`. KQBNvKR's native file
is **18,577,168 bytes** with **13,531,664,160 native index slots**. KNNvKNN uses
**1,013,968 bytes** and **1,546,346,340 stored slots**: color symmetry and repeated
pieces reduce its native storage domain. These are native streams, including
invalid fillers and capture-reconstructable codes, not a count of legal boards.
Uniform/stratified compression samples are explicitly labelled; their sizes are
not extrapolated into a complete custom archive.

The hardware model uses H100 **SXM**, 80 GB and 3.35 TB/s. Dense FP16 is 989.5
TFLOPS; dense INT8 is 1,979 TOPS. The 1,979 FP16 / 3,958 INT8 figures advertised
with the sparsity footnote must not be used as dense rates. This model does not
describe an H100 PCIe card, whose bandwidth/performance differ. Rust and cuTile
are the implementation target; the GPU source build is currently blocked by
absent CUDA dependencies in the development environment.

For the GPU working layout, retain canonical nonadjacent king pairs but pad the
four other piece coordinates to 64 each. Both turns give
`2 * 462 * 64^4 = 15,502,147,584` candidate cells. This is about 14.6% more than
the published native slot count. Do not filter duplicate diagonal-stabilizer
orientations out of quiet fibers: they provide harmless redundancy that keeps
the local move graph regular. King moves normalize the king pair and transform
all piece coordinates; output uses the compact canonical native/storage index.
The legality mask excludes collisions and invalid check states. This layout
index now has a CPU Rust ranker and an independent closure fixture generator.
All 4,096 king-pair inputs, 4,566 legal quiet transitions including 946 king
moves, and 2,053 random/boundary ID roundtrips are checked. Nonking transitions
change one base-64 digit and turn directly; king transitions transform the four
other coordinates through small lookup tables. This verifies the padded quiet
index, not capture dependencies or an end-to-end GPU solve. Native storage uses
its own maintained compact ranker; the padded index is not a Syzygy file index.

The bounded WDL working model uses about **28.3 GB**: current/next two-bit state,
validity, static legal-move and capture-boundary masks, one nonwinning-child bit
for each dynamic move family, lower-table allowance and tile scratch. A full
16-bit all-side DTM buffer adds about 31.0 GB; a single-side 16-bit DTZ buffer
adds about 15.5 GB. Distances exceeding the narrow representation require exact
overflow escapes or a wider/streamed stage. Never silently saturate them.

The exact quiet operator fixes five pieces and varies the sixth over 64 squares.
For a nonking mover, physical source validity and destination own-king safety
factor from geometry. Captures are excluded and processed in the boundary path.
The CPU fixture suite independently checks the full legal relation, including
positions in check, for knight, bishop, rook and queen fibers with both colors.
It compares every one of the 64×64 adjacency cells in each of 320 fibers.

For knights, `A` is the **same 64×64 matrix** for all board skeletons. Each column
is a different real fiber with its own masks and child state. Compute

`B[j,g] = safe[j,g] * (1 + 9*[child != Win] + 81*[child == Loss])`.

The INT8 values are 0, 1, 10 or 91. Each knight row has at most eight destinations,
so `Z=A*B` has base-9 digits equal to legal, nonwinning and losing child counts;
the maximum is 728. INT32 accumulation is exact. FP16 inputs with FP32
accumulation also represent these values/sums exactly, providing a comparison
kernel. FP8 E4M3 **cannot** represent 91 exactly; do not substitute it here.
Decode the three digits, apply source validity, and pack output masks. This is
the reference encoding. The faster iterative encoding uses **0/16/17** instead:
`B = safe * (16*[child != Win] + [child == Loss])`, so `Z & 15` detects a losing
child and `Z != 0` detects a nonwinning child, with maximum sum 136. It eliminates
division/remainder and emits only two dynamic bitplanes. Cache move existence
separately; zero output alone must never classify stalemate as loss. FP8 E4M3
cannot represent 17 exactly. No large floating-point state array or per-fiber
adjacency is stored in HBM.

A further variant uses **0/8/9**, exactly representable in FP8 E4M3:
`B = safe * (8*[child != Win] + [child == Loss])`. Now `Z & 7` detects one to
seven losing destinations. Eight losses carry, but uniquely give `Z == 72`,
because eight is the maximum degree. Therefore `win = ((Z & 7) != 0 || Z == 72)`
and `nonwinning = (Z != 0)` are exact. The CPU Rust suite enumerates all 65,536
assignments of illegal/winning/unresolved/losing children at a degree-eight
source, including this carry and no-legal-move cases. The GPU comparison source
uses this common encoding for INT8/INT32, FP16/FP32 and FP8/FP32 candidates.
It requests ordinary accumulation, without the optional reduced-precision fast
FP8 mode. Input conversion and bounded FP32 integer sums are exact in the CPU
model; compiled device arithmetic remains untested. The listed FP8 peak is an
optimistic ceiling, not the achieved throughput of strict accumulation.

The knight matrix also has no more than two nonzeros per contiguous four-column
group. Thus a 2:4 sparse representation can preserve it exactly by filling unused
slots with zeros. The current cuTile API/source benchmark does **not** implement
that sparse path. Treat it as a cuda-oxide/PTX experiment, not an achieved 2×
speedup. Its usefulness depends on actual pack/reduction/register overhead.

The alternative knight operator is eight masked bit shifts. Rook/bishop/queen
operators use occluded fills with 1/2/4-square stages, stopping at **physical
blockers**. Destination safety masks filter endpoints; an unsafe intermediate
square does not block a ray. This produces all 64 source predicates without a
64×64 legal-move matrix. These integer stencils are serious competitors to
tensor cores, despite their lower advertised arithmetic throughput.

Keep the knight coordinate as the contiguous bitboard axis. Other piece axes
require a 64×64 bit transpose within a tile; fuse those transposes with loads and
packed candidate output when possible. If layouts are materialized instead,
charge each transpose's full read/write traffic to the roofline. Pack/scatter to
one owning output word per CTA; do not hide uncoalesced or atomic traffic in an
ideal one-byte-per-row calculation.

Use monotone WDL values and idempotent dirty tile notifications. Each family
caches whether a parent has a nonwinning child. Stale cached `true` delays a
loss; cached `false` remains sound because proven child wins never revert.
Uncomputed family contributions stay conservative. A parent wins when any legal
child loses; it loses only when it has a legal move and every family's nonwinning
mask is false. Checkmate and stalemate are distinct seeds. Unresolved cycles are
draws only after complete global quiescence, including delivered boundaries and
notifications. Duplicate messages must not decrement counters twice.

Dense dirty tiles can batch independent knight columns for MMA. Sparse late
frontiers and king/capture ranking use SIMT. Pick the crossover from measured
whole-tile times, not TFLOPS. For larger jobs, shard owners persist exact boundary
states and epochs; termination must include all in-flight work. No R2 GET should
occur inside the iterative GPU fixed point: decode the needed lower tables into
resident device caches before timing the solve.

Those device dependencies must contain final, correctly probed legal-board WDL
values, or an exact compatible GPU capture prober. Raw native stored codes have
fillers and capture reconstruction rules; copying their signs directly into
generation seeds would be wrong. Decode/prepare by blocks, using constant and
grammar-run expansion, rather than calling a scalar CPU prober for every slot.

Precompute king-move transforms: each canonical king pair, turn and neighboring
king square maps to a new pair plus one of eight board transforms. The compact
table and 8×64 square-transform LUT fit in cache; rank the four other coordinates
after applying the transform. Actual attack/capture legality remains mandatory.
Asynchronous WDL commitment depth is only a proof rank, not necessarily optimal
DTM. Compute exact distances in an ordered/layered or separately validated stage.

The resident packed INT8 and packed bitboard knight operators both have an
optimistic **0.75 bytes per source row** footprint in the two-output-word
benchmark. HBM therefore caps either at about **4.47 trillion source rows/s**;
the packed INT8 dense arithmetic cap is about 15.5 trillion rows/s. Neither bound
includes legality-mask construction, transpose, scalar packing instructions,
spills, king moves, capture ranks or global convergence. A source row for one
piece is **not** a whole-position evaluation.

An illustrative INT32 issue budget is 16.75 trillion lane instructions/s, using
the H100's 64 INT32 versus 128 FP32 lanes/SM and the 67 TFLOPS FP32 peak's two
operations/FMA convention. At 8/16/32/64 scalar lane instructions per source row,
this gives 2.09/1.05/0.52/0.26 trillion rows/s before occupancy and other limits.
These are sensitivity cases, not measured kernel instruction counts. Packing
via warp ballots may beat weighted 64-bit tile reductions; inspect the emitted
code and measure it. A bitboard stencil performs many operations on 64 source
rows at once. Sparse MMA or more tensor TOPS cannot remove scalar packing costs.

For runtime planning, use `T = N * equivalent_sweeps / full_position_rate`, then
add initialization, lower-table preparation, rule/distance stages, checking,
compression and durable output. At 100 equivalent sweeps over the padded layout,
1 / 5 / 20 billion **complete** position evaluations per second would take about
25.8 / 5.2 / 1.3 minutes of compute. At 20 sweeps those figures are five times
smaller. These are sensitivity scenarios, not measured or assured throughputs;
the required round count is unknown until the complete solver runs.

Instrument every phase with CUDA events and host wall time. Preflight the actual
GPU variant, toolkit, driver, memory, clocks and power settings. Warm JIT before
operator timing, report JIT separately, and use a working set substantially above
L2. Record useful source rows, real legal edges, dirty-tile density, physical HBM
bytes, spills, and boundary traffic. Measure INT8, FP16, packed stencils and their
layout costs on identical inputs. FMA/TOPS and nominal logical bytes are not
Nsight counters. The cuTile benchmark validates all results against the CPU
oracle before timing, but has not executed on an H100 here.

For durable storage, preserve **native Syzygy compression as the baseline**.
At 16,384 positions/page, our old 32-byte fixed records alone would need 26.4 MB
for KQBNvKR, already exceeding its entire 18.6 MB native file. Avoid paying a
per-logical-page header on extremely compressible materials. The new native
range format adds one SHA-256 hash per **physical 16 KiB** tile (about 0.195%),
with hash pages authenticated by the catalog. The compiled Rust filesystem
implements exact random reads through shared caches and the maintained native
Syzygy rank/probe library, preserving captures and published five-class rules.

Use physical hash/range pages and warmed header/grammar/index caches. Keep source
objects content-addressed; reject corrupt data and hash pages before decoding.
The wrapper retains the immutable `.rtbw/.rtbz` bytes rather than transcoding
them. A new GPU-generated result may initially use exact streaming packed/Zstd
blocks, then compete with Syzygy-style compression on complete files. The final
representation selector must count authenticated indexes, dictionaries, routing
and rule metadata as well as payload. A larger compute-friendly working tensor
does not imply a larger final object collection.

The measured current six-piece mirror lists 365 WDL files totaling **72.83 GB**
and 365 DTZ files totaling **87.39 GB** (decimal). Together, about 160.23 GB of
native data; physical hash indexes add about 0.2%, plus catalogs. At the listed
R2 Standard $0.015/GB-month this is roughly **$2.41/month before the free storage
allowance and billing rounding**, excluding requests/API compute. This is the
existing six-piece collection, not an eight-piece estimate. Cold reads can still
require several metadata/data tiles and capture dependencies; warm-cache rates
from the small test must not be extrapolated to a full collection.

The lower three/four/five-piece directories add 396,105,616 WDL bytes and
587,852,304 DTZ bytes. The listed full collection through six pieces totals
**161,209,573,952 native bytes**, or about **161.52 GB with physical hashes**,
plus small catalogs. For a constant whole-month stored quantity this rounds
to 162 GB-month: **$2.43/month before free storage**, or **$2.28 if the full
10 GB allowance is available**. This excludes operations and API compute.

`native_six_validation.json` records 1,005 independently checked FENs, cold/warm
passes at 64 MiB and 1 MiB cache budgets, native WDL/DTZ fixtures, eight concurrent
batch sessions, evictions, missing dependencies and corrupt/short HTTP resources.
In the 64 MiB trace, batching reduces cold origin reads **565 -> 448** at the
same 8,927,936 fetched bytes; both modes have zero warm origin reads. Batching
adds local CPU work, so scalar remains preferable for a cheap local store.
The corresponding marginal R2 Standard origin cost, *if those cold trace miss
rates occurred at scale*, is about **$0.202 -> $0.160 per million positions**,
before free operations and billing rounding. These are trace-dependent origin
costs, not live cloud service prices. Actual request/compute cost and cache
locality still require measurement; see `six_lookup_costs.json`.
