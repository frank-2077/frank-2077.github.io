---
layout: default
title: 关于
lang: zh
permalink: /about/
alt_url: /en/about/
---

<div class="intro">
  <p>我是<strong>何思如</strong>，南方科技大学<strong>电子信息硕士</strong>，与中国科学院深圳先进技术研究院联合培养，方向是<strong>大模型推理与 AI Infra</strong>。</p>

  <p>
    实习经历集中在推理系统与算子这两层：在<strong>面壁智能（ModelBest）</strong>基础模型中心参与 MiniCPM-o 4.5、VoxCPM2 的多模态推理运行时研发；
    在<strong>上海光羽芯辰</strong>负责自研端侧 NPU 上大模型 Decode 阶段的算子优化，参与 Decode 阶段 Megakernel 整图融合。
  </p>

  <ul>
    <li><strong>CUDA / Triton 算子开发</strong> —— 访存优化、Tensor Core、确定性计算；Memory-Bound 算子的 cycle 级优化</li>
    <li><strong>推理引擎搭建与优化</strong> —— 从零搭建到性能调优，覆盖 TP 分片、KV Cache 管理、MoE 执行重构、CUDA Graph、投机解码</li>
    <li><strong>大模型 Serving 与框架贡献</strong> —— 熟悉 vLLM、Megatron、llama.cpp/ggml 内部机制，向 vLLM-Omni 提交稳定性与性能 PR</li>
    <li><strong>分布式与训推一致性</strong> —— Batch Invariance 与确定性归约，8×H100 上训推 logprob bitwise 一致</li>
    <li><strong>性能剖析与工具链</strong> —— Nsight Compute、PyTorch Profiler、Perf、GDB；CI 与自动化工作流</li>
  </ul>

  <p>技术栈：C++ / CUDA / Triton / Python / PyTorch / vLLM / Megatron / llama.cpp(ggml) / NCCL / Docker / CMake / Git</p>

  <p>论文：<strong>SteadyFlow-Edge</strong>（第一作者，ACAI）、<strong>RT-EdgeDetect</strong>（第一作者，APPT）</p>

  <hr>

  <p>
    邮箱：<a href="mailto:{{ site.email }}">{{ site.email }}</a><br>
    GitHub：<a href="https://github.com/frank-2077">frank-2077</a>
  </p>
</div>
