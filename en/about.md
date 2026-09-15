---
layout: default
title: About
lang: en
permalink: /en/about/
alt_url: /about/
---

<div class="intro">
  <p>I'm <strong>SIRU HE</strong> (何思如), an <strong>LLM Inference / AI Infra engineer</strong>, currently doing a Master's in Electronic Information at SUSTech, jointly trained with SIAT-CAS.</p>

  <p>
    My focus is the low-level engineering of AI. As models grow and have to run fast and cheaply
    on both phones and servers, someone needs to optimize the layers closest to the hardware:
    GPU kernels, inference engines, and distributed systems.
  </p>

  <ul>
    <li><strong>Kernels</strong> — CUDA / Triton operator development, memory-access and Tensor Core optimization</li>
    <li><strong>Inference systems</strong> — building inference engines, KV Cache, CUDA Graph, speculative decoding</li>
    <li><strong>On-device deployment</strong> — NPU optimization and streaming inference for multimodal LLMs</li>
    <li><strong>Train-serve consistency</strong> — reproducible train vs. inference results across distributed clusters</li>
  </ul>

  <p>Tech stack: C++ · CUDA · Triton · Python · PyTorch · vLLM · llama.cpp/ggml</p>

  <hr>

  <p>
    Email: <a href="mailto:{{ site.email }}">{{ site.email }}</a><br>
    GitHub: <a href="https://github.com/frank-2077">frank-2077</a><br>
    Homepage: <a href="https://frank-2077.github.io">frank-2077.github.io</a>
  </p>
</div>
