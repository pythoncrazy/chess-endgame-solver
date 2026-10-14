# Compression experiments

This file records the compression experiments for the `.tbz` sections.
Each entry gives the question, the method, the results and the decision.
All sizes are in bytes.

## Test set

The test set has five pawnless tables from `results/all5-v5`.

| Table | Stored positions | Type |
|---|---|---|
| KQvKR | 5,242,880 | four pieces, many wins |
| KRvKB | 5,242,880 | four pieces, many draws |
| KQBvKR | 335,544,320 | five pieces, many wins |
| KRNvKQ | 335,544,320 | five pieces, mixed results |
| KBNvKN | 335,544,320 | five pieces, mixed results |

The baseline is the v4 file: zstd level 19 on each 256 KiB block.
A WDL block has 16384 words. A rank block has 4096 words.

## E1: adaptive models with one context

Question: Can a simple context model do better than zstd on the stored values?

Method: The tool `rust/src/bin/tb_entropy.rs` decodes a table.
It then finds the ideal code length of an adaptive model on each section.
The model uses Krichevsky-Trofimov counts for each context.
Each model starts again at each block, as a block coder must.
The tool does not code illegal positions or positions that are not stored.

Contexts for WDL:

- The left and upper black-king squares of the same word.
- The same black-king square in the words at index distance 1, 8, 64 and 512.
- The two upper diagonal squares.

Contexts for rank:

- The WDL value of the position and the rank of the left square.
- The difference to the first known rank of the left square, the upper square and word distance 1.

Results (best single model against zstd):

| Table | WDL zstd | WDL model | Rank zstd | Rank model |
|---|---|---|---|---|
| KQvKR | 31,926 | 27,124 | 483,333 | 384,435 |
| KRvKB | 70,062 | 58,560 | 43,098 | 38,548 |
| KQBvKR | 1,567,469 | 1,639,772 | 14,117,600 | 15,054,841 |
| KRNvKQ | 11,064,158 | 10,213,500 | 11,861,506 | 7,990,929 |
| KBNvKN | 5,592,225 | 4,270,205 | 5,690,352 | 4,443,538 |

A choice of the best model for each block gives almost no gain.

Decision: One context is not enough. KQBvKR is worse than zstd.
The next step is a mixer of many contexts.

## E2: binary context mixer

Question: How much does a mixer of many contexts save?

Method: The mixer codes each value as a sequence of binary decisions.
This is the method of the PAQ coders.

- WDL: three decisions. They are "draw", "loss" and "black-to-move win of rank 1".
- Rank: the coder predicts the rank from the left square, the upper square or word distance 1. The first known value is the prediction.
- Rank: the decisions are "difference is zero", the sign, the length in unary and the low bits.
- Each model keeps an adaptive probability for each context and decision.
- The mixer adds the model outputs in the logistic domain. It learns its weights online.
- WDL uses eight models. Rank uses seven models.
- An SSE stage then refines the result. SSE is secondary symbol estimation.

The default settings are a learn rate of 0.005 and a count limit of 255.

Results with the v4 block sizes:

| Table | WDL zstd | WDL mixer | Rank zstd | Rank mixer |
|---|---|---|---|---|
| KQvKR | 31,926 | 13,245 | 483,333 | 297,237 |
| KRvKB | 70,062 | 38,939 | 43,098 | 17,100 |
| KQBvKR | 1,567,469 | 899,545 | 14,117,600 | 10,493,552 |
| KRNvKQ | 11,064,158 | 7,471,247 | 11,861,506 | 5,732,412 |
| KBNvKN | 5,592,225 | 3,470,769 | 5,690,352 | 2,858,586 |
| Total | 18,325,840 | 11,893,745 | 32,195,889 | 19,398,887 |

The total for all sections is 31,292,632 bytes. The zstd total is 50,521,729 bytes.
The mixer is 38% smaller. It is smaller on each section of each table.

Effect of the settings on the total:

| SSE | Learn rate | Count limit | Rank predictor | Total |
|---|---|---|---|---|
| yes | 0.005 | 255 | first known | 31,292,632 |
| yes | 0.005 | 255 | median (LOCO-I) | 31,359,945 |
| no | 0.005 | 255 | first known | 31,414,698 |
| no | 0.02 | 255 | first known | 31,478,430 |
| no | 0.05 | 255 | first known | 32,741,453 |
| no | 0.02 | 60 | first known | 31,465,951 |
| no | 0.02 | 1023 | first known | 31,521,968 |

Effect of the block size (no SSE, learn rate 0.02, count limit 255):

| WDL words | Rank words | Total | Change |
|---|---|---|---|
| 65536 | 16384 | 31,080,589 | -1.3% |
| 16384 | 4096 | 31,478,430 | 0 |
| 4096 | 1024 | 33,701,938 | +7.1% |
| 1024 | 256 | 36,742,243 | +16.7% |

Speed: one CPU thread codes approximately 11 million positions each second.
Thus one full WDL block of 16384 words takes approximately 100 ms to decode.
The zstd decoder takes less than 1 ms for one block.

Decision: Make a real coder from the mixer.

- The coder must have a fast decoder for single probes.
- Smaller blocks decrease the probe time but increase the size.
- A GPU can code thousands of blocks in parallel for the packer.

## E3: run contexts for WDL

Question: Do run lengths help the WDL models?

Method: The walk counts three runs inside the block.

- The run of values that are equal to the previous coded value.
- The run of values that are equal to the value at word distance 1.
- The run of values that are equal to the value of the left square.

Three new models use these runs in buckets 0, 1, 2-3, 4-7 and up to 64.

Results for the adaptive mixer with the v4 block sizes:

| Table | WDL before | WDL after | Change |
|---|---|---|---|
| KQvKR | 13,245 | 12,106 | -8.6% |
| KRvKB | 38,939 | 36,635 | -5.9% |
| KQBvKR | 899,545 | 807,568 | -10.2% |
| KRNvKQ | 7,471,247 | 7,030,928 | -5.9% |
| KBNvKN | 3,470,769 | 3,311,138 | -4.6% |

The adaptive total for all sections is now 30,597,262 bytes.
This is 39% smaller than zstd.

Decision: Keep the run contexts.

## E4: semi-static mixer for small blocks

Question: Can a fixed model replace the adaptive model?

An adaptive model has two problems.

- A decoder must decode the block from the start. A probe is thus slow.
- Each block is a sequence. The GPU can only use one thread for each block.

A fixed model gives each decision a probability from its contexts only.
Thus all decisions of a block are independent.
A GPU can find all probabilities in parallel. An interleaved rANS coder can then code them.
Small blocks do not lose model data, because the model is in the header.

Method:

1. Count the bits of each context and decision over the full table.
2. Keep a 12-bit probability for each context with TB_PRUNE bits or more. Other contexts give the mixer a zero input.
3. Train the mixer weights for two epochs on each eighth block.
4. Find the code length with the fixed model and fixed weights.
5. Find the header size: zstd level 19 on a presence bitmap and the 12-bit values.

A WDL block has 256 words (16384 positions). A rank block has 64 words (4096 positions).
These blocks are 64 times smaller than the v4 blocks.

Results with TB_PRUNE 64:

| Table | WDL data | WDL header | Rank data | Rank header |
|---|---|---|---|---|
| KQvKR | 25,451 | 20,270 | 316,183 | 38,150 |
| KRvKB | 63,004 | 9,685 | 26,009 | 4,374 |
| KQBvKR | 1,714,464 | 50,266 | 11,832,153 | 86,040 |
| KRNvKQ | 9,278,925 | 241,102 | 5,964,389 | 113,561 |
| KBNvKN | 4,343,456 | 18,049 | 3,130,388 | 54,769 |

| TB_PRUNE | Data | Header | Total |
|---|---|---|---|
| 1 | 36,574,818 | 1,118,612 | 37,693,430 |
| 16 | 36,610,929 | 860,828 | 37,471,757 |
| 64 | 36,694,422 | 636,266 | 37,330,688 |

The total is 26% smaller than zstd. The adaptive mixer is 39% smaller.
The fixed model is almost equal to the adaptive model on rank.
It is worse on WDL, because the WDL data changes inside a table.

A test with v4 block sizes shows the cause of the WDL loss.
With these blocks, the fixed WDL model of KQBvKR is 1,578,638 bytes. The adaptive model is 807,568 bytes.
Thus the block size is not the cause. The cause is the fixed model.

Decision: Use the fixed model for the rank section.
Find a better fixed model for WDL, or use the adaptive model for WDL.
