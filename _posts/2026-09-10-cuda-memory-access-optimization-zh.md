---
layout: post
title: "CUDA 访存优化：从合并访存到双缓冲流水"
date: 2026-09-10 21:30:00 +0800
lang: zh
permalink: /2026/09/cuda-memory-access-optimization/
categories: cuda
tags: [cuda, gpu, sgemm, 访存优化]
---

这篇是我做 **CUDA SGEMM 优化**（Ampere / Hopper）时的一点梳理：为什么访存会成为瓶颈，以及从「合并访存」到「memcpy_async 双缓冲」每一级优化背后的原理。目的是给出一条可复用的排查与优化路径，而不是一堆经验之谈。

## 先从算术强度说起

GPU 算得快，但**喂数据喂不快**。用「算术强度」（FLOPs/Byte）衡量一次访存能换回多少计算：

```text
算术强度 = 总 FLOPs / 总访存字节数
```

4096×4096 的矩阵乘法，朴素实现每个元素读一次、做一次乘加，算术强度只有 0.125 FLOPs/Byte——大部分时间都花在等内存上。而优化到最后能把 SGEMM 推到 32 FLOPs/Byte，靠的不是算得更快，而是**让同样的访存换回更多计算**。

## 第一层：合并访存

GPU 的内存事务以 **32 字节的 sector** 为单位。如果一个 warp 里的 32 个线程访问的地址落在同一个 sector 内，就只发一次事务；散得越开，发的事务越多。

规则很直接：**让相邻线程访问相邻地址。**

```c
// 差：线程按列访问，跨 stride，一个 warp 被打散到多个 sector
float v = A[row * N + col];

// 好：线程连续访问
float v = A[row * N + (col + threadIdx.x)];
```

用 Nsight Compute 看 Warp 内的 `Sectors/Request`，把它从 32 降到 1，单这一步就能提速约 8 倍。这是收益最高、也最该最先做的一步。

## 第二层：Shared Memory 分块

全局内存再合并也还是慢。把要反复用的数据搬进 **Shared Memory（SMEM）**，让同一块 tile 被读多次：

```text
全局内存 --搬一次--> SMEM --读多次--> 寄存器计算
```

分块的关键不是 tile 越小越好，而是**复用**。2D Block Tiling 让每个搬进 SMEM 的元素被消费多次，从而摊薄全局访存。

## 第三层：向量化与 Bank Conflict

`float4` 一次搬 128 bit，直接减少事务数量：

```c
float4 *As = reinterpret_cast<float4*>(&smem[idx]);
float4 a = As[i];
```

但 SMEM 有 32 个 bank，同一时刻多个线程访问同一个 bank 会串行化。解决办法是 **padding**——每行末尾补一列，把访问模式错开：

```c
__shared__ float smem[BLOCK_DIM][BLOCK_DIM + 1];  // +1 消除 bank conflict
```

## 第四层：异步流水

到这一步为止，搬运和计算还是串行的：「搬完再算」。用 `cuda::memcpy_async` 可以提前发出**下一块** tile 的搬运，让它与**当前块**的计算重叠：

```text
迭代 i:   算 tile[i]   |  搬 tile[i+1]
迭代 i+1: 算 tile[i+1] |  搬 tile[i+2]
```

配合 `cuda::barrier` 做到达/等待同步，就形成了一条双缓冲流水线。在 Hopper 上这一步可以升级为 **TMA（Tensor Memory Accelerator）**——由单线程发起 bulk copy，连搬运的指令开销都省掉了。

## 一条可复用的路径

1. 看 `Sectors/Request`——修合并访存。
2. 看 SMEM 复用率——上分块。
3. 向量化加载——修 bank conflict。
4. 最后上异步流水，重叠搬运与计算。

顺序很重要：**先合并，再复用，最后重叠。** 反过来做往往会掩盖真正的瓶颈。

> 这套路径在 RTX 3090 上把 4096×4096 的 SGEMM 从 548ms 压到 6.4ms，达到 cuBLAS 的 93%。Hopper 上 TMA 与 TF32 WMMA 那部分我后面另写一篇。
