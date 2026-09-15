---
layout: default
title: About
lang: en
permalink: /en/about/
alt_url: /about/
---

<div class="intro">
  <p>I'm <strong>SIRU HE</strong>, a Master's student in Electronic Information at SUSTech, jointly trained with SIAT-CAS, focused on <strong>LLM inference and AI infrastructure</strong>.</p>

  <p>
    My internship experience sits at the inference-system and kernel layers: at <strong>ModelBest</strong>
    I worked on the multimodal inference runtime for MiniCPM-o 4.5 and VoxCPM2; at
    <strong>Shanghai Guangyu Xinchen</strong> I optimized LLM Decode-stage operators on a self-developed
    edge NPU and worked on Decode-stage megakernel fusion.
  </p>

  <ul>
    <li><strong>CUDA / Triton kernels</strong> — memory access, Tensor Core, deterministic computation; cycle-level optimization of memory-bound operators</li>
    <li><strong>Inference engines</strong> — from building an engine to tuning it: TP sharding, KV Cache management, MoE execution, CUDA Graph, speculative decoding</li>
    <li><strong>Serving & framework contribution</strong> — internals of vLLM, Megatron, llama.cpp/ggml; stability and performance PRs to vLLM-Omni</li>
    <li><strong>Distributed & train-serve consistency</strong> — Batch Invariance and deterministic reductions; bitwise-identical logprobs on 8×H100</li>
    <li><strong>Profiling & tooling</strong> — Nsight Compute, PyTorch Profiler, Perf, GDB; CI and automation workflows</li>
  </ul>

  <p>Tech stack: C++ / CUDA / Triton / Python / PyTorch / vLLM / Megatron / llama.cpp (ggml) / NCCL / Docker / CMake / Git</p>

  <p>Publications: <strong>SteadyFlow-Edge</strong> (first author, ACAI), <strong>RT-EdgeDetect</strong> (first author, APPT)</p>

  <hr>

  <p>
    Email: <a href="mailto:{{ site.email }}">{{ site.email }}</a><br>
    GitHub: <a href="https://github.com/frank-2077">frank-2077</a>
  </p>
</div>
