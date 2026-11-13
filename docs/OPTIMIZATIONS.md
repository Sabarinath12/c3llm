# Optimization log

How c3llm went from about 1 token per second to about 36, in the order it happened, with what was measured at each step and what did not work.

## Summary

Generation speed for Llama 3.2 1B (`Llama-3.2-1B-Instruct-Q4_0.gguf`, 773 MB) on a Ryzen 5 5500U laptop: 6 cores, 12 threads, AVX2, 16 GB dual-channel DDR4-3200.

| # | Change | 1 thread | All threads |
|---|---|---|---|
| 0 | First working forward pass, scalar | 1.1 tok/s | n/a |
| 1 | Float SIMD kernels and a thread pool | 2.7 | 9.3 |
| 2 | Correct CPU build flag | 5.5 | 17.7 |
| 3 | Thread pool rewritten to spin | 5.5 | 17.8 |
| 4 | Integer kernels | 6.4 | 22.5 |
| 5 | Half-float lookup table | 10.5 | 30.8 |
| 6 | Leaner integer kernels | 11.1 | 30.6 |
| 7 | General parallel loop, attention and SiLU threaded | 11.1 | 37.4 |
| | `llama.cpp` b12322 on the same file | | about 37 |

On full 200-token replies through the compare page the final figure is 35.2 to 36.1 tok/s, against 37.1 to 38.1 for `llama.cpp`.

The last column is the bench command at the best thread count for that stage (6 threads until stage 7, then 10).

## How it was measured

- **`c3llm bench <model> [threads]`** runs 32 forward passes after 4 warm-up passes and prints milliseconds per token, split by tensor type, plus the kernel speed on a slice small enough to stay in cache.
- **Standalone micro-benchmarks** timed kernel variants on one in-cache row, to compare designs without the rest of the engine.
- **The compare page** (`node compare\server.js`) runs the same prompt through both engines, one after the other.
- **Cool-down pauses.** This is a 15 W laptop chip. Back-to-back runs differed by 10% or more from heat alone, so later comparisons alternated builds with a 10 to 12 second pause before each run.

One fact shaped everything after stage 2: with one core busy the chip runs at about 3.9 GHz, and with all cores busy at about 3.0 GHz. Six threads therefore give about 3.5 times one thread at best, not 6.

## Stage 0: the starting point

The first forward pass was written for correctness:

- **Zero-copy weights.** The model file is memory-mapped and tensors are read in place.
- **KV cache.** Each token only computes attention against cached keys and values.
- **No logits for prompt tokens.** The output projection (128,256 rows) is skipped for every prompt token but the last.
- **Scalar kernels.** Q4_0 had a direct loop; Q4_1 and Q6_K expanded each block to floats and then multiplied.
- **One thread.**

Result: 1.1 tok/s.

## Stage 1: float SIMD and threads

- **Vector kernels.** Dot products were rewritten with C3's `float[<16>]` vectors: unpack the nibbles, convert to float, multiply by the activations.
- **Thread pool.** Worker threads shared the rows of each matrix product, claiming 64 rows at a time under a mutex and sleeping on a condition variable between products.
- **Q6_K needed its own kernel.** The first attempt left Q6_K on the expand-then-multiply path, and the output layer alone took 1.25 seconds per token. A direct vector kernel fixed that.

Result: 2.7 tok/s on one thread, 9.3 on six. `llama.cpp` measured 37.4 on the same machine, about 4 times faster.

## Stage 2: the build flag

Inspecting the generated assembly showed no FMA instructions at all, and 64 calls to the C library's scalar `fmaf`. The build used `--x86cpu=native`, which on this Zen 2 CPU only enabled AVX1.

Building with `--x86cpu=avx2-v2` instead doubled the speed with no code change. `build.ps1` now always passes it.

Result: 5.5 tok/s on one thread, 17.7 on six.

## Stage 3: thread pool with atomics and spinning

Six threads were only 3.2 times faster than one, so the pool was the first suspect. Matrix products arrive tens of microseconds apart, which is about how long a sleeping thread takes to wake.

- **Atomic row claims** replaced the mutex.
- **Spin-waiting.** Idle workers spin on the job counter and only sleep after about a millisecond with no work.

Result: no measurable change (56.1 ms per token against 58). The diagnosis that followed was more useful than the change:

- **Page faults** were flat after warm-up, so the mapped file was not being re-faulted.
- **Scaling was sub-linear from two threads up**, even with the data in cache.
- **Clock speed** explained it: 3.9 GHz with one core busy, 3.0 GHz with six.

So the limit was compute per token, and the lever was a cheaper kernel.

## Stage 4: integer kernels

`llama.cpp` rounds the activations to 8-bit integers and multiplies integers. The same idea here:

- **Activations rounded once per matrix product.** Each block of 32 values (256 for Q6_K) gets its own scale so its largest value maps to the top of the integer range.
- **Multiply-add as plain vector code.** The kernel needs the AVX2 instruction `vpmaddwd`, which multiplies 16 pairs of 16-bit values and adds neighbours. C3 cannot bind the intrinsic on Windows, because vector arguments are passed by pointer and LLVM rejects the declaration. Writing the operation out works, because LLVM recognises the shape:

```c3
macro int[<8>] madd(short[<16>] a, short[<16>] b)
{
	int[<16>] m = (int[<16>])a * (int[<16>])b;
	return swizzle(m, 0, 2, 4, 6, 8, 10, 12, 14) + swizzle(m, 1, 3, 5, 7, 9, 11, 13, 15);
}
```

- **Q4_1 got a real kernel** by sharing the Q4_0 loop. It had been on the slow fallback path and went from 16 ms to 4.4 ms per token.

Result: 22.5 tok/s on six threads. But the Q4_0 kernel was no faster on one thread than the float version, which led to the next stage.

## Stage 5: half-float lookup table

A micro-benchmark of kernel variants found where the time went:

| Variant | Billion weights per second |
|---|---|
| Kernel as written | 8.1 |
| Same, without converting the block's half-float scale | 15.7 |
| Integer work only | 27 |

Every block starts with a 16-bit float scale. Converting it with the CPU's `vcvtph2ps`, one value at a time, cost as much as all the rest of the block.

The fix is a 65,536-entry table, filled once at start-up, which is also what `llama.cpp` does:

```c3
float[65536] half_table;
fn float f16(char* p) @inline => half_table[mem::load((ushort*)p, 1)];
```

Converting a whole row's scales 8 at a time before the main loop was just as fast, but needed buffers and chunking, so the table won on simplicity.

Result: 10.5 tok/s on one thread, 30.8 on six.

## Stage 6: leaner kernels

More micro-benchmarking of the Q4 kernel:

| Variant | Billion weights per second |
|---|---|
| Stage 5 kernel | 16.3 |
| Activations stored as 16-bit, offset subtracted as integers | 18.8 |
| Same, with fused multiply-add and two accumulators | 21.1 |

What changed:

- **Activations stored as 16-bit.** `vpmaddwd` reads them straight from memory instead of widening 8-bit values on every block.
- **Format offsets precomputed.** Q4_0 weights mean `nibble - 8` and Q6_K weights mean `value - 32`. Instead of subtracting from every weight, the kernels multiply the unsigned values and subtract a correction computed once per matrix product from the activation totals.
- **Two accumulators.** Blocks are processed in pairs into separate fused multiply-add accumulators, so one block does not wait for the previous one.
- **Q6_K rework.** Group scales are widened once per block and applied with a 16-bit multiply before the multiply-add. Its in-cache speed went from 11.0 to 13.4 billion weights per second.

Result: 11.1 tok/s on one thread, but six threads did not move. Both large tensor types were now streaming at the same 24.5 GB/s, which pointed at memory.

## Stage 7: memory, and the remaining single-threaded code

A build that only read the weights, with almost no computation, measured the machine's ceiling:

| Threads | Read speed |
|---|---|
| 1 | 16.4 GB/s |
| 3 | 28.5 GB/s |
| 6 | 30.0 GB/s |

At 30 GB/s the 773 MB file takes about 26 ms to read, so about 37 tokens per second is the limit for this model on this machine. `llama.cpp` sits at that limit.

What got the engine there:

- **One general parallel loop.** The pool now runs any function over a range, not just matrix rows.
- **Attention and SiLU on all threads.** Both had been single-threaded. Attention work grows with the length of the conversation, which is why long replies had been slower than the bench suggested. Everything outside matrix products fell from about 2.2 ms to 0.9 ms per token.
- **Contiguous shares with stealing.** Each thread starts on its own contiguous run of rows, so each core reads memory as one steady stream, and helps the others once its share is done.
- **Yield while waiting.** With every logical CPU spinning, a worker the OS had descheduled mid-chunk could be starved by the threads waiting for it. At 12 threads that collapsed throughput to about 15 tok/s. Waiting threads now give up their CPU every 256 spins.
- **Default thread count.** All logical CPUs but two (10 here) measured best: 26.9 ms per token, against 29.0 at 6 and 27.3 at 12.
- **Finer rounding at no cost.** With 16-bit storage the activations use 15 bits for the 4-bit formats and 10 for Q6_K, where `llama.cpp` uses 8. The limits are chosen so every integer sum stays within 32 bits.

Result: 37.4 tok/s in the bench, 35.2 to 36.1 on 200-token replies.

## What did not work

- **`--x86cpu=native`.** It silently selected AVX1 on this CPU. Check the assembly, not the flag.
- **Binding LLVM intrinsics.** Declaring `llvm.x86.avx2.pmadd.wd` as an external function compiles, then fails in LLVM because C3 passes vectors by pointer on Windows.
- **The byte multiply-add `vpmaddubsw`.** It would save two or three instructions per block. LLVM only emits it for a saturating pattern, and the optimizer removes the saturation because it can prove the values are small. Hiding the constants from the optimizer did not bring the pattern back; both attempts ran at about 6 billion weights per second.
- **Software prefetch.** Prefetching 1 KB ahead cost about 10%, and 4 KB ahead about 3%.
- **Spinning in the thread pool, by itself.** Wake-up latency was not the bottleneck at that stage. Spinning only mattered later, and needed the yield to be safe.
- **`vcvtph2ps` for block scales.** See stage 5.
- **Reducing each block to a scalar before scaling.** The horizontal sum costs more than converting 8 lanes to float.

## Accuracy

The self-test checks each integer kernel against the float reference on inputs that survive rounding exactly, so the kernels must agree closely.

Against `llama.cpp` with greedy sampling, replies start identical and usually drift apart after a few hundred characters; of five prompts, one stayed identical for its full length. The two engines round activations differently, and greedy sampling turns any difference at a near-tie into different wording. Both stay coherent. With the original float kernels, the one prompt tested matched for 80 tokens.

## What is left

- **Prompt reading** is still one token at a time. `llama.cpp` batches prompt tokens and reads prompts about 4 times faster.
- **Generation** is within about 5% of the memory limit. Going faster means reading fewer bytes per token, which means a smaller or more heavily quantized model.
- **The thread default** was tuned on one machine. Run the bench command with different thread counts on other hardware.
