---
layout: default
title: 关于
lang: zh
permalink: /about/
alt_url: /en/about/
---

<div class="intro">
  <p>我是<strong>何思如</strong>，一名<strong>大模型推理 / AI Infra 工程师</strong>，目前在南方科技大学读电子信息硕士，与中国科学院深圳先进技术研究院联合培养。</p>

  <p>
    我的方向是 AI 的底层工程。模型越来越大，还要在手机和服务器上都跑得又快又省，
    就需要有人去优化最贴近硬件的部分：GPU 算子、推理引擎和分布式系统。
  </p>

  <ul>
    <li><strong>算子与内核</strong> —— 写 CUDA / Triton 算子，优化访存与 Tensor Core 计算</li>
    <li><strong>推理系统</strong> —— 从零搭建推理引擎，做 KV Cache、CUDA Graph、投机解码</li>
    <li><strong>端侧部署</strong> —— 多模态大模型的端侧 NPU 优化与流式推理</li>
    <li><strong>训推一致性</strong> —— 让训练和推理在分布式集群上结果可复现</li>
  </ul>

  <p>常用技术栈：C++ · CUDA · Triton · Python · PyTorch · vLLM · llama.cpp/ggml</p>

  <hr>

  <p>
    邮箱：<a href="mailto:{{ site.email }}">{{ site.email }}</a><br>
    GitHub：<a href="https://github.com/frank-2077">frank-2077</a><br>
    主页：<a href="https://frank-2077.github.io">frank-2077.github.io</a>
  </p>
</div>
