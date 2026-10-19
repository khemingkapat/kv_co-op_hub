# Mastering LLM Techniques: Inference Optimization (NVIDIA Technical Blog)

## 0. Plain-English TL;DR (What It Is & Why It Matters)

- **The Big Picture:**
  This technical guide by NVIDIA breaks down the fundamental mechanics of Large Language Model (LLM) serving on GPUs. It explains why running an LLM divides into two completely different phases (reading the prompt vs. generating text), why generating text is strictly bound by memory speed rather than raw compute math, and how systems techniques like Key-Value (KV) caching, PagedAttention, FlashAttention, and in-flight batching attempt to squeeze maximum throughput out of GPU hardware.

- **Core Concepts Explained Simply:**
  - **Prefill Phase (Reading the Prompt):** The GPU ingests the entire user prompt at once. Because all input words are known simultaneously, the GPU can run massive parallel matrix multiplications ($GEMM$). This fully saturates the GPU's compute cores (compute-bound).
  - **Decode Phase (Generating New Words):** The model produces text autoregressively, one word at a time. For every single token generated, the GPU must fetch all previous model weights and historical Key/Value states from memory to calculate a single matrix-vector ($GEMV$) step. The GPU's compute cores finish almost instantly and spend most of their time waiting for data to travel across memory buses (memory-bandwidth bound).
  - **KV Cache Memory Footprint:** Instead of re-reading and recalculating previous tokens at every step, the GPU stores past Keys and Values in VRAM. This avoids redundant math, but causes VRAM consumption to grow linearly with every new token generated and every concurrent user served.

---

## 1. Key Takeaways for Our Project

- **What This Means for Our 8–16 GB Consumer GPUs:**
  - Standard serving optimizations like **FlashAttention** and **PagedAttention** are necessary, but they do **not** solve the consumer memory wall alone:
    - *FlashAttention* speeds up memory access by tiling operations in fast on-chip SRAM, but it still requires the full uncompressed KV cache to reside in GPU VRAM.
    - *PagedAttention* stops memory waste caused by fragmentation (unallocated virtual pages), but it does not compress or shrink the actual volume of tokens.
  - On a typical consumer card (8 GB RTX 4060 or 16 GB RTX 4090 cap):
    - A 7B–8B model in 16-bit precision takes $\approx 14\text{--}16\text{ GB}$ just for weights, leaving zero room for KV cache. Even a downscaled 3B–4B model occupies $6\text{--}8\text{ GB}$, leaving only $4\text{--}8\text{ GB}$ of VRAM headroom.
    - Under long contexts ($32\text{K}$) or shared SME concurrency (10–15 users), the KV cache alone demands tens of gigabytes. Without 3-axis co-optimization (selective quantization, CPU offloading, and transfer pipelining), consumer GPUs instantly crash with Out-Of-Memory (OOM) errors.

- **Formulas & Rules of Thumb to Keep:**
  - **Per-Token KV Cache Size (Bytes):**
    $$\text{KV Bytes per Token} = 2 \times \text{num\_layers} \times (\text{num\_heads} \times \text{dim\_head}) \times \text{precision\_bytes}$$
    *Plain English:* Multiply by 2 because we store both Keys and Values. For modern models with Grouped-Query Attention (GQA), $\text{num\_heads} \times \text{dim\_head}$ uses the number of *KV heads* ($H_{KV}$), not Query heads.
  - **Total KV Cache Size across a Batch:**
    $$\text{Total KV Cache} = \text{batch\_size} \times \text{sequence\_length} \times \text{KV Bytes per Token}$$
    *Plain English:* Memory scales directly with how many users you serve and how long the conversation gets.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 12 in `manual_reviews.md` correctly noted that decoding is memory-bandwidth bound and that FlashAttention, GQA, and PagedAttention reduce memory overhead and boost throughput.
  - *Refinement from Document:* While GQA reduces cache size at model design time and PagedAttention stops fragmentation waste, neither technique dynamically offloads or compresses existing cache entries when context scales to 32K on a single consumer GPU. This reinforces the necessity of our project's policy layer.

---

## 2. System & Architectural Analysis

- **Core Contribution:**
  Provides the reference hardware-software baseline for LLM inference on NVIDIA architectures, establishing why autoregressive generation underutilizes GPU compute and surveying runtime optimizations available in modern serving engines (TensorRT-LLM, vLLM).

- **Architectural Bottlenecks Addressed:**
  1. *Decode Arithmetic Intensity:* The operational intensity of the decode step is very low ($\approx 1\text{ FLOP/byte}$ of weight/KV read). Latency is strictly governed by memory bandwidth (e.g., $272\text{ GB/s}$ on an RTX 4060 vs. $1008\text{ GB/s}$ on an RTX 4090).
  2. *Static Over-Provisioning:* Older engines statically pre-allocated memory for the maximum possible sequence length (e.g., 2048 tokens) per request, leaving massive unused memory bubbles.
  3. *Static Batching Inefficiencies:* Traditional batching forces all sequences in a batch to wait for the single longest completion to finish. In-flight (continuous) batching evicts completed requests iteration-by-iteration to insert new requests dynamically.

- **Relevance to `kv_bench` Baseline Work Packages (`WP-1` & `WP-2`):**
  - Sets the theoretical expectations for our profiling harness: Time-To-First-Token (TTFT) reflects prefill compute saturation, while Time-Per-Output-Token (TPOT) reflects memory-bandwidth and PCIe transfer bottlenecks.
  - Confirms our pinned model memory calculations in [`docs/MODELS.md`](../../MODELS.md) and provides the ground truth formulas used in our proposal's Section 2.1.
