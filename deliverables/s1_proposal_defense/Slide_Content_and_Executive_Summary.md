# Senior Project Proposal — Slide Content & Technical Executive Summary

**Project Title:** Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs  
**Deck File:** [KV-Cache Co-optimization.pdf](file:///c:/Users/Newol/cpe-semester-2-2026/CPE401-Project-I/slides/KV-Cache%20Co-optimization.pdf) · [Canva Link](https://www.canva.com/design/DAHWrgV7N5M/edit)  
**Total Slides:** 16 Slides  
**Advisors:** Dr. Aye Hninn Khine · Dr. Boonyarit Changaival  
**Department:** Computer Engineering, Faculty of Engineering, KMUTT  
**Academic Year:** 2026–2027 (B.E. 2568–2569)  
**Project Members:**
1. Kantawich Limwilai (Student ID: 66070503406)
2. Khem Ingkapat (Student ID: 66070503408)
3. Sirawat Sanboonsong (Student ID: 66070503456)
4. Run Prajunban (Student ID: 66070503480)

---

## 📌 2-Minute Executive Summary

* **The Problem:** In autoregressive Large Language Model (LLM) decoding, Key-Value (KV) cache memory scales **linearly** with sequence length. For 32K context lengths, the KV-cache alone requires **16 to 32 GB of VRAM**, exceeding the physical memory ceiling of consumer GPUs (e.g., NVIDIA RTX 4060 8 GB / 16 GB).
* **Current Limitations (Siloed Optimizations):**
  * *Eviction (Token Dropping):* Deletes tokens permanently, causing irreversible quality degradation in long multi-turn interactions.
  * *Quantization (Compression):* Compresses cache to 2/4-bit, but cache size still grows linearly and causes numerical precision loss if applied uniformly.
  * *Offloading (CPU Swapping):* Moves tensors to system RAM, but bottlenecks severely on PCIe transfer bandwidth (causing compute stall bubbles).
* **Our Proposed Solution (Co-Optimization):**
  A dynamic, decoupled **3-Axis Policy Layer** that coordinates:
  1. **Structural Axis:** Evaluates token importance via rolling attention scores to keep critical tokens intact.
  2. **Spatial Axis:** Selectively quantizes low-priority tokens to INT4 and offloads them to Host DRAM (saving ~75% VRAM).
  3. **Temporal Axis:** Asynchronously prefetches and delegates dequantization to idle CPU cores, hiding PCIe transfer latency behind active GPU decode steps.
* **Target Outcome:** Enable unmodified open-weight models (Llama-3, Qwen-2.5, Gemma-3, Mistral) up to 32K context on consumer GPUs with 30–40% higher throughput and 95–98% quality retention.

---

# 📑 Slide-by-Slide Content Breakdown

---

### Slide 01 — Project Title & Overview
* **Tag:** `PROPOSAL DEFENSE / TITLE`
* **Topic:** Introduction to the Senior Project Proposal

#### 📖 Slide Explanation & Core Concept
This opening slide introduces the senior engineering research project conducted under the Department of Computer Engineering at KMUTT. The title highlights the exact technical intersection: **Co-optimizing KV-Cache management** specifically tailored for **LLM inference on consumer-grade GPUs**.

#### 🎯 Key Takeaways
* **Clear Research Scope:** The project focuses on inference (decoding), not training or fine-tuning.
* **Target Hardware:** Consumer GPUs (8–16 GB VRAM, specifically NVIDIA RTX 4060).
* **Core Value Proposition:** Running long-context LLMs locally without out-of-memory crashes.

---

### Slide 02 — The KV-Cache Bottleneck
* **Tag:** `01 / PROBLEM BACKGROUND`
* **Topic:** The fundamental memory scaling barrier in Transformer decoding

#### 📖 Slide Explanation & Core Concept
Autoregressive LLM generation requires caching past key and value tensors at every Transformer layer to avoid re-evaluating preceding tokens at quadratic cost $O(S^2)$. However, this creates a severe memory footprint:
$$\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem} \times S_{\text{context}} \times B_{\text{batch}}$$

As context grows from 4K to 32K and 64K tokens, the cache alone consumes 16 to 32 GB of VRAM. Because typical consumer hardware provides only 8 to 16 GB total VRAM, the system crashes immediately with Out-Of-Memory (OOM).

#### 🎯 Key Takeaways
* In modern long-context inference, **Memory Capacity and Memory Bandwidth are the primary bottlenecks, not raw Compute (TFLOPS)**.
* Even modest 7B–8B models cannot run long contexts on consumer cards without aggressive cache management.

#### 📚 Key Terminology
| Term | Meaning |
| :--- | :--- |
| **KV-Cache** | Cache storing Key and Value projection states of past tokens across Transformer layers |
| **Autoregressive Decoding** | Generating output one token at a time, where each new token depends on all previous tokens |
| **OOM (Out-Of-Memory)** | Runtime crash occurring when CUDA memory allocation exceeds physical GPU VRAM |

---

### Slide 03 — Real-World Motivation: Consumer Hardware Floor
* **Tag:** `02 / MOTIVATION`
* **Topic:** Practical deployment reality and the hidden cost of local LLM gateways

#### 📖 Slide Explanation & Core Concept
The demand for private, local LLMs has surged (e.g., OpenClaw, Ollama, LM Studio reaching hundreds of thousands of users). Users assume running models locally is "free," but encounter a harsh **Hardware Floor**. Official deployment documentation explicitly states that handling 64K context requires at least 8–16 GB VRAM, but warns: *"This requirement does not guarantee fit or speed."*

When users submit long multi-turn queries, document reviews, or large codebases, naive runtimes freeze or crash. Existing isolated fixes simply shift the bottleneck rather than resolving it.

#### 🎯 Key Takeaways
* Local LLM adoption is blocked by consumer hardware limitations.
* A co-optimized runtime bridges the gap between commercial long-context needs and consumer affordability.

---

### Slide 04 — Four Objectives, One Constraint
* **Tag:** `03 / PROJECT OBJECTIVES`
* **Topic:** Measurable project goals under a rigid single-GPU budget

#### 📖 Slide Explanation & Core Concept
The project balances four simultaneous objectives under the strict constraint of consumer hardware (RTX 4060 8 GB / 16 GB):
1. **Retain Quality:** Maintain $\ge 95\text{--}98\%$ task accuracy and negligible perplexity delta compared to uncompressed FP16 baselines.
2. **Reduce Footprint:** Fit 32K+ context window within 8–16 GB VRAM.
3. **Hide Transfer:** Completely overlap host-to-device PCIe data transfers with GPU decode computation (zero latency bubbles).
4. **Raise Throughput:** Deliver 30–40% higher token generation throughput than uncompressed baselines under multi-session concurrency.

#### 🎯 Key Takeaways
* Achieving all four goals requires an adaptive runtime rather than static trade-offs.

---

### Slide 05 — Literature Review: Three Dominant Paradigms
* **Tag:** `04 / LITERATURE REVIEW`
* **Topic:** Analysis of existing research and their isolated trade-offs

#### 📖 Slide Explanation & Core Concept
Existing literature addresses KV-cache memory through three isolated paradigms:
1. **Eviction (Token Dropping) — e.g., H2O, StreamingLLM:**
   * *Mechanism:* Retains only top-attended tokens ("heavy hitters") and attention sinks, discarding the rest.
   * *Fatal Risk:* Permanent, unrecoverable information loss. In multi-turn dialogues, dropped historical facts cannot be recalled.
2. **Quantization (Compression) — e.g., QServe, KIVI:**
   * *Mechanism:* Compresses cache tensors to 2-bit or 4-bit integers.
   * *Fatal Risk:* Cumulative numerical error distortion across long sequences; cache size still grows linearly.
3. **Offloading (Tiered Storage) — e.g., FlexGen, DeepSpeed:**
   * *Mechanism:* Swaps inactive cache blocks to host DRAM.
   * *Fatal Risk:* PCIe bus bandwidth bottleneck creates "latency bubbles" (GPU stalls while waiting for data transfer).

#### 🎯 Key Takeaways
* No single existing paradigm solves the consumer GPU problem alone without severe trade-offs.

---

### Slide 06 — Research Gap: The Missing Orchestration Layer
* **Tag:** `05 / RESEARCH GAP`
* **Topic:** The Co-Design Opportunity across three axes

#### 📖 Slide Explanation & Core Concept
Prior systems fail because they treat KV-cache optimization as an isolated 1-dimensional problem. A complete system must answer three questions simultaneously:
* **Structural:** *Which* tokens are critical enough to preserve in full precision?
* **Spatial:** *Where* should tokens reside (VRAM vs DRAM), and in *what* format (FP16 vs INT4)?
* **Temporal:** *When* should transfers occur so the GPU never idles?

Optimizing any single axis in isolation degrades another. The **Co-Optimization Gap** represents the opportunity to orchestrate all three dimensions together.

---

### Slide 07 — Proposed Methodology: 3-Axis Co-Optimization Framework
* **Tag:** `06 / PROPOSED METHODOLOGY`
* **Topic:** The central Policy Coordinator framework

#### 📖 Slide Explanation & Core Concept
The proposed framework unites the three axes under an active **Policy Coordinator**:
* **Structural Axis (Attention Priority):** Dynamically ranks tokens based on attention weights and persistence.
* **Spatial Axis (Multi-Tier Placement):** Distributes tokens across GPU VRAM (FP16 for critical tokens) and Host DRAM (INT4 for secondary tokens).
* **Temporal Axis (Pipelined Scheduling):** Asynchronously streams and prefetches cache blocks over PCIe via dual CUDA streams.

#### 🎯 Key Takeaways
* The Policy Coordinator acts as the central brain making dynamic, real-time cache placement decisions.

---

### Slide 08 — System Architecture: Decoupled Layers
* **Tag:** `07 / SYSTEM ARCHITECTURE`
* **Topic:** 3-layer decoupled architectural design

#### 📖 Slide Explanation & Core Concept
To ensure modularity and clean engineering, the architecture is decoupled into three coordinated layers:
* **L1 Priority Engine:** Observes runtime attention dynamics and token recency to score importance and protect critical context.
* **L2 Cache Format Manager:** Manages multi-precision tensor representation (FP16 $\leftrightarrow$ INT4), block residency, and virtual allocation.
* **L3 Transfer Scheduler:** Coordinates asynchronous DMA transfers between CPU host RAM and GPU VRAM over the PCIe bus.

#### 🎯 Key Takeaways
* Decoupling allows independent testing and optimization of priority logic, quantization kernels, and memory streaming pipelines.

---

### Slide 09 — Dynamic Attention-Based Priority Engine
* **Tag:** `08 / PRIORITY ENGINE`
* **Topic:** 4-step execution pipeline and Zero-Loss Guardrail

#### 📖 Slide Explanation & Core Concept
At each decode step, the Priority Engine executes a 4-step pipeline:
1. **Observe:** Reads current attention weight distribution and token elapsed steps.
2. **Score:** Calculates a composite score combining instantaneous attention with historical persistence:
   $$\text{Priority}(t) = \alpha \cdot \text{Attention}(t) + \beta \cdot \text{Persistence}(t)$$
3. **Protect (Zero-Loss Guardrail):** Applies mandatory protection rules:
   * **Attention Sinks:** Initial 1–4 prompt tokens (BOS) permanently pinned.
   * **System Prompt:** Core instructions permanently pinned.
   * **Recent Window:** Last $W$ tokens kept in full FP16.
4. **Act:** Unprotected tokens are selectively quantized to INT4 and scheduled for host DRAM offloading.

#### 🎯 Key Takeaways
* The **Zero-Loss Guardrail** guarantees that core reasoning and attention sinks are never degraded or evicted.

---

### Slide 10 — Asynchronous Offload & Latency Hiding
* **Tag:** `09 / ASYNCHRONOUS OFFLOAD`
* **Topic:** Compute-Transfer Overlapping via Dual CUDA Streams

#### 📖 Slide Explanation & Core Concept
PCIe Gen3/Gen4 bandwidth (~16–31.5 GB/s) is significantly slower than internal GPU VRAM bandwidth (~272 GB/s on RTX 4060). To prevent GPU stalls:
* **Dual CUDA Streams:**
  * `Stream 0 (Compute):` Executes FlashAttention-2 and next-token decoding for token $t$.
  * `Stream 1 (Memory DMA):` Concurrently prefetches and transfers quantized cache blocks for token $t+1$ from CPU to GPU.
* **CPU Compute Delegation:** While the GPU is computing token $t$, idle host CPU cores use vectorized AVX-512 instructions to dequantize INT4 blocks back to FP16 in pinned memory. When transferred to GPU, they are immediately ready for computation without GPU dequantization overhead.

$$\text{Total Step Latency} = \max(T_{\text{Compute}}, T_{\text{PCIe Transfer}}) \approx T_{\text{Compute}}$$

#### 🎯 Key Takeaways
* Transfer overhead is completely hidden behind GPU compute time.

---

### Slide 11 — Reproducible Testbed Specifications
* **Tag:** `10 / EVALUATION PLAN`
* **Topic:** Accessible, reproducible experimental hardware & software matrix

#### 📖 Slide Explanation & Core Concept
* **Hardware:**
  * *Local Workstation:* NVIDIA GeForce RTX 4060 (8 GB VRAM), 16–32 GB Host RAM, PCIe Gen4.
  * *Cloud/Cluster:* NVIDIA RTX 4090 node capped at 16 GB VRAM allocation.
* **Software:** PyTorch 2.x, CUDA 12.x, custom C++/CUDA kernel extensions integrated with vLLM PagedAttention runtime.
* **Target Models (Unmodified Weights):**
  * Llama-3 (8B), Qwen-2.5 (1.5B / 3B / 7B), Mistral (7B).
* **Workloads:**
  * Context scaling: 4K $\rightarrow$ 8K $\rightarrow$ 16K $\rightarrow$ 32K tokens.
  * Synthetic multi-turn SME enterprise chat traces (10–15 concurrent user sessions).

---

### Slide 12 — Evaluation Plan: Baselines & Target Metrics
* **Tag:** `11 / BASELINES & TARGET METRICS`
* **Topic:** Comparative baselines and quantitative target indicators

#### 📖 Slide Explanation & Core Concept
The system is compared against 5 distinct baselines:
1. *Native vLLM (Uncompressed FP16):* Standard reference.
2. *FlexGen:* State-of-the-art CPU-GPU offloading baseline.
3. *H2O (Heavy-Hitter Oracle):* Pure eviction baseline.
4. *QServe / KIVI:* Pure quantization baseline.
5. *ShadowKV:* Recent KV offload system keeping compressed keys on GPU and offloading values.

#### 🎯 Quantitative Target Indicators
* **Peak VRAM:** 50–75% reduction under long contexts (fitting 32K context in 8 GB VRAM).
* **Throughput:** 30–40% speedup over naive offloading.
* **Language Quality:** $\le 0.1$ perplexity delta; $\ge 95\text{--}98\%$ LongMemEval score retention.
* **Latency:** Sub-100ms Time-Per-Output-Token (TPOT).

---

### Slide 13 — Team Responsibilities & Ownership
* **Tag:** `12 / TEAM CONTRIBUTIONS`
* **Topic:** Individual ownership and collaborative delivery

#### 📖 Slide Explanation & Core Concept
Clear division of engineering responsibility:
* **Sirawat (New):** Priority Engine implementation, Memory Offloading pipeline, Local RTX 4060 testbed adaptation.
* **Run:** Problem analysis, Literature review, Objectives definition, Task quality evaluation.
* **Khem:** System architecture, Cache formats, Benchmarking harness (`kv_bench`), Slurm HPC testbed.
* **Kantawich (Bluebird):** System design, Technical documentation, Project coordination, Milestone deliverables.
* **Shared Ownership:** System integration, end-to-end benchmarking, academic paper writing, final defense.

---

### Slide 14 — Project Timeline: Semester 1 (Aug–Dec 2026)
* **Tag:** `13 / PROJECT TIMELINE`
* **Topic:** Semester 1 Roadmap (Research, Profiling & Prototype Co-Design)

#### 📖 Key Milestones
* **Aug–Sep:** Literature survey, testbed setup, initial uncompressed baseline profiling.
* **Oct–Nov:** 3-Axis policy design, Priority Engine prototype, memory offloading simulation.
* **Dec:** Milestone sanity check, Semester 1 Progress Report & Proposal Defense.

---

### Slide 15 — Project Timeline: Semester 2 (Jan–May 2027)
* **Tag:** `13 / PROJECT TIMELINE`
* **Topic:** Semester 2 Roadmap (Runtime Integration, Stress Testing & Thesis)

#### 📖 Key Milestones
* **Jan–Feb:** Integration with production vLLM runtime, correctness verification.
* **Mar–Apr:** Comprehensive 32K stress benchmarking, ablation studies, kernel tuning.
* **May:** Thesis document completion, open-source artifact release, Final Senior Project Defense.

---

### Slide 16 — Key References & Academic Defense Compendium
* **Tag:** `14 / REFERENCES & Q&A`
* **Topic:** Foundational papers and defense Q&A preparation

#### 📖 Selected Academic References
1. **FlexGen:** Sheng et al., *"FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"* (ICML 2023).
2. **H2O:** Zhang et al., *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"* (NeurIPS 2023).
3. **StreamingLLM:** Xiao et al., *"Efficient Streaming Language Models with Attention Sinks"* (ICLR 2024).
4. **PagedAttention / vLLM:** Kwon et al., *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* (SOSP 2023).
5. **ShadowKV:** Sun et al., *"ShadowKV: KV Cache Approximation with Value-Only Offloading"* (2024).

---

## 🛡️ Technical Q&A Defense Guide

### Q1: Why not just use vLLM as-is?
**Answer:** vLLM's PagedAttention solves internal memory fragmentation within VRAM by allocating memory in non-contiguous pages. However, vLLM inherently assumes that the entire active KV-cache fits into physical VRAM. On consumer GPUs with only 8–16 GB VRAM, long-context requests (16K–32K) immediately exceed physical memory capacity and cause fatal CUDA Out-Of-Memory (OOM) crashes. Our system builds a tiered, adaptive offload and compression layer on top of vLLM's architecture, allowing long contexts to spill gracefully into host memory without modifying model weights.

### Q2: Doesn't 4-bit quantization degrade model reasoning accuracy?
**Answer:** Uniform quantization (compressing all tokens to INT4) causes significant perplexity degradation and numerical drift over long sequences. Our approach uses **Selective Quantization** governed by the **Zero-Loss Guardrail**:
1. Critical tokens (system prompts, attention sinks, and immediate sliding-window context) are permanently preserved in full **FP16 precision in GPU VRAM**.
2. Only low-attention, non-critical background tokens are quantized to INT4 before being staged to host memory.
Empirical studies in literature confirm that retaining the top 10–20% of tokens in FP16 maintains $\ge 95\text{--}98\%$ of original model accuracy.

### Q3: PCIe bandwidth is notoriously slow — can asynchronous transfers really keep up?
**Answer:** Yes, via two complementary techniques:
1. **INT4 Quantization** reduces data transfer volume across the PCIe bus by ~75% (4 bits vs 16 bits per element).
2. **Dual CUDA Stream Pipelining** overlaps data transfers with active compute. While the GPU is executing the attention kernel for token $t$, the next token's cache blocks ($t+1$) are already being prefetched in parallel over the PCIe DMA stream.
Furthermore, by having idle CPU cores perform vectorized dequantization before transmission, the GPU executes without compute stall bubbles.

### Q4: What is the specific scientific novelty of this project?
**Answer:** Prior literature addresses KV-cache memory in silos: Eviction methods (H2O, StreamingLLM) drop tokens and suffer permanent context loss; Offloading systems (FlexGen) suffer PCIe saturation; Quantization methods (QServe) still scale linearly.  
Our novelty is the **dynamic, closed-loop co-optimization policy across all three dimensions simultaneously**:
* Dynamically determining *which* tokens matter (Structural),
* Selecting *where* they reside and at what bit-width (Spatial), and
* Scheduling *when* to transfer and decompress them to hide latency (Temporal).

---

*CPE401 Senior Project · KMUTT Computer Engineering · Academic Year 2026–2027*
