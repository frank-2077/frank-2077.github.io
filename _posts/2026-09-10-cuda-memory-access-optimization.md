---
layout: post
title: "CUDA Memory Access: From Coalescing to Double-Buffered Pipelines"
date: 2026-09-10 21:30:00 +0800
categories: cuda
tags: [cuda, gpu, sgemm, memory]
---

Notes from optimizing a **CUDA SGEMM** kernel (Ampere / Hopper): why memory access becomes the bottleneck, and the reasoning behind each optimization step — from coalescing to `memcpy_async` double buffering. The goal is a reusable path for diagnosing and fixing memory-bound kernels, not a pile of folklore.

## Start with arithmetic intensity

GPUs compute fast but are **slow to feed**. Arithmetic intensity (FLOPs/Byte) measures how much computation each byte of memory traffic buys you:

```text
arithmetic intensity = total FLOPs / total bytes of memory traffic
```

A naive 4096×4096 matrix multiply reads each element once and does one multiply-add — an intensity of just 0.125 FLOPs/Byte, meaning most of the time is spent waiting on memory. Pushing SGEMM to 32 FLOPs/Byte isn't about computing faster; it's about **getting more computation out of the same memory traffic**.

## Layer 1: coalesced access

GPU memory transactions move in **32-byte sectors**. If the 32 threads of a warp hit addresses inside the same sector, that's one transaction; if they're scattered, it's 32.

The rule: **adjacent threads should touch adjacent addresses.**

```c
// Bad: threads stride by column, scattering a warp across many sectors
float v = A[row * N + col];

// Good: threads read contiguously
float v = A[row * N + (col + threadIdx.x)];
```

Watch `Sectors/Request` in Nsight Compute: dropping it from 32 to 1 alone can give roughly an 8× speedup on that step. It's the highest-return change and the one to do first.

## Layer 2: shared memory tiling

Even perfectly coalesced global memory is slow. Bring the data you reuse into **shared memory (SMEM)** so a single tile is read many times:

```text
global memory --load once--> SMEM --read many times--> register computation
```

Tiling isn't about making tiles as small as possible — it's about **reuse**. A 2D block tile lets each element loaded into SMEM be consumed many times, amortizing the global traffic.

## Layer 3: vectorization and bank conflicts

`float4` loads 128 bits at once, cutting the transaction count directly:

```c
float4 *As = reinterpret_cast<float4*>(&smem[idx]);
float4 a = As[i];
```

But SMEM has 32 banks, and threads hitting the same bank serialize. The fix is **padding** — one extra column per row, offsetting the access pattern:

```c
__shared__ float smem[BLOCK_DIM][BLOCK_DIM + 1];  // +1 kills bank conflicts
```

## Layer 4: asynchronous pipelining

So far every load is "fetch, then compute" — serialized. With `cuda::memcpy_async` you issue the *next* tile's transfer early and overlap it with the *current* tile's compute:

```text
iter i:   compute tile[i]   |  load tile[i+1]
iter i+1: compute tile[i+1] |  load tile[i+2]
```

Pair it with `cuda::barrier` for arrive/wait synchronization to form a double-buffered pipeline. On Hopper this can be upgraded to **TMA (Tensor Memory Accelerator)**: a single thread issues the bulk copy, removing even the instruction overhead of the transfer.

## A reusable path

1. Check `Sectors/Request` — fix coalescing.
2. Look at SMEM reuse — add tiling.
3. Vectorize loads — fix bank conflicts.
4. Finally, add async pipelining to overlap transfer and compute.

Order matters: **coalesce first, then reuse, then overlap.** Doing it backwards tends to hide the real bottleneck.

> On an RTX 3090 this path took a 4096×4096 SGEMM from 548ms down to 6.4ms — 93% of cuBLAS. I'll write up the Hopper TMA and TF32 WMMA side another time.
