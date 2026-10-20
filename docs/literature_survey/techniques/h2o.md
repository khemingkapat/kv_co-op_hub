# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  During language model generation, the Key-Value (KV) cache grows linearly with every newly generated word and every concurrent user. For large models and long sequences, this temporary memory quickly exceeds the size of the model itself (for example, a 30-billion-parameter model generating 1,024 words across 128 batches eats 180 GB of VRAM just for the cache, easily crashing GPU memory). Simply dropping older words with a sliding window destroys accuracy because the model loses critical context from the prompt. Meanwhile, textbook optimal caching (Belady's algorithm) cannot be used in real-world LLM serving because no one knows which past words the model will need in future generation steps.

- **How It Solves It (Step-by-Step in Plain Words):**
  1. **Before / During Prefill (Spotting Heavy Hitters):**
     When reading the initial prompt, H2O tracks how much attention each word receives. The authors discover a striking natural phenomenon: over 95% of attention in modern language models concentrates onto a tiny handful (~5%) of tokens, termed "Heavy Hitters". If the prompt exceeds the cache limit, H2O immediately identifies and saves these critical heavy hitters alongside a small window of the most recent words, discarding non-essential tokens right away.
  2. **Where the Data Lives (Storage & Memory Split):**
     H2O strictly caps the KV cache size in GPU memory (e.g., keeping only 20% of the full sequence). The cache budget is split into two halves: one half stores the cumulative heavy hitters that have consistently received high attention, and the other half stores a sliding window of the most recent words. Tokens that fall outside both pools are permanently removed from GPU memory. In systems with CPU offloading, shrinking the cache footprint allows much larger batches to stay resident on the GPU instead of being bottlenecked by slow PCIe transfers.
  3. **During Decode (Greedy Token Eviction on the Fly):**
     Whenever the model generates a new word, it calculates attention over the cached tokens and adds these new scores to a running cumulative tally. The newest word is placed into the recent sliding window. The oldest word in that recent window is then pushed out to compete with the heavy hitters: whichever token across the candidate pool has accumulated the lowest total attention score across history is evicted on the spot. To prevent memory fragmentation and slow GPU memory allocations, H2O pre-allocates a static memory block and overwrites evicted entries in-place.

- **How It Compares to Our Project:**
  H2O is the pioneer of importance-based cache pruning. However, H2O permanently deletes evicted tokens from memory, leading to irreversible loss of facts if a user asks a question about an evicted detail later in a multi-turn conversation. In our 3-axis project, we do not permanently discard tokens; instead, we use H2O-style importance scoring as a priority signal to decide which tokens remain in fast GPU memory at full precision and which tokens are compressed (quantized to 4-bit) or staged to host CPU memory.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Slashes KV cache memory usage by 5x to 10x (matching full-cache quality with only 10% to 20% of the cache); boosts generation throughput on offloading systems by up to 3x over FlexGen and up to 29x over DeepSpeed ZeRO-Inference and Hugging Face Accelerate; reduces generation latency by 1.1x to 1.9x for long sequences.
  - **What You Pay:** Permanent loss of evicted tokens (once dropped, forgotten context cannot be recovered); requires tracking cumulative attention vectors across every attention head, which conflicts with closed-source fused attention kernels (like standard FlashAttention) unless attention weights or approximations are explicitly exposed.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **The Dual-Pool Balance (Heavy Hitters + Recent Window):** Neither heavy hitters alone nor recent tokens alone are sufficient. In the paper's ablation studies, keeping only recent tokens causes the model to lose the plot (accuracy collapses by 20% to 37%), while keeping only heavy hitters disrupts immediate grammar and sentence structure. An even split between accumulated high-attention tokens and a local sliding window is the key combination that preserves full-cache performance.
  2. **Zero-Copy In-Place Tensor Replacement:** Dynamically reallocating memory or moving tensors around in GPU VRAM during decoding ruins inference latency. H2O avoids this by pre-allocating a static buffer for the budget: the recent window is managed as a circular ring queue, and evicted heavy-hitter slots are overwritten in-place with newly promoted tokens without invoking PyTorch memory allocators.
  3. **Pruning and Quantization Naturally Compound:** The authors found that H2O token pruning combines seamlessly with 4-bit KV quantization, retaining baseline accuracy (e.g., 78.80% on PiQA vs 78.51% full cache) while compounding throughput improvements. This provides strong empirical validation for our project's co-optimization strategy (Structural eviction + Precision quantization).
  4. **Initial Token Bias Foreshadowing Attention Sinks:** The authors observed that a disproportionately high concentration of heavy hitters consistently appears at the very beginning of sequences (the prompt start tokens). This early observation directly foreshadowed the discovery of attention sinks in StreamingLLM.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Permanent Deletion vs. Multi-Turn SME Workloads:** On our target workloads (10–15 concurrent small-business users doing document question-answering and prolonged multi-turn chat), permanent token eviction is dangerous. When a user asks a question referring back to a specific entity or number mentioned 10 turns earlier that wasn't scored as a global heavy hitter, the model hallucinates or fails.
  - **FlashAttention Compatibility Bottleneck:** Standard high-performance inference engines (vLLM, FlashInfer, FlashAttention-2/3) rely on fused online softmax kernels that compute attention without ever writing the attention matrix to GPU memory. Accumulating token attention scores at every step requires either hooking into attention weights or running separate scoring approximations, which adds kernel overhead.
  - **Evaluated on Server Hardware:** The paper's benchmarks were run on an NVIDIA T4 (16 GB Turing) and an NVIDIA A100 (80 GB SXM4). On consumer hardware (an 8 GB RTX 4060 with PCIe Gen 4 x8, or a 16 GB RTX 4090 cap), host-device PCIe bandwidth is a major bottleneck; any CPU offloading strategy must be paired with quantization to avoid stalling the generation pipeline.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our manual notes correctly identified the central concept: creating an eviction policy that retains heavy hitter tokens (~5% of the cache) to reduce KV cache size, and noted evaluation on OPT, LLaMA, GPT-NeoX, and WikiText-103.
  - *Clarification from PDF:* The team note mentioned: *"I don't get it very much because I skipped the implementation and took the ideas... need to dig into the implementation on it for more"*. The exact implementation mechanism is now verified: H2O splits the cache into $K$ heavy-hitter slots and $K$ recent slots ($2K$ total). It maintains a cumulative attention score vector $\tilde{o}_i = \tilde{o}_{i-1} + o_i$ updated at every decode step. Tokens in the recent window are managed by a circular ring buffer; as each token exits the recent window, H2O greedily evicts the token with the lowest cumulative attention score in $O(1)$ in-place tensor replacement. Furthermore, the paper proves this greedy eviction achieves a $(1 - 1/e)$ approximation ratio under a dynamic submodular optimization framework.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Zhenyu Zhang (UT Austin), Ying Sheng (Stanford), Tianyi Zhou (UCSD), Tianlong Chen (UT Austin), Lianmin Zheng (UC Berkeley), Ruisi Cai (UT Austin), Zhao Song (Adobe Research), Yuandong Tian (Meta AI FAIR), Christopher Ré (Stanford), Clark Barrett (Stanford), Zhangyang Wang (UT Austin), Beidi Chen (Meta AI FAIR / CMU)
- **Publication / Venue & Year:** 37th Conference on Neural Information Processing Systems (NeurIPS 2023), December 2023. (arXiv:2306.14048v3, Dec 2023)
- **Primary Axis:** Structural (Token-level cumulative attention score eviction)
- **Secondary Interacting Axis:** Spatial (Integrated into FlexGen memory hierarchy with CPU offloading and KV quantization)
- **Official Codebase / Repository:** [https://github.com/FMInference/H2O](https://github.com/FMInference/H2O)
- **Baseline Implementation Stack:** Built on top of FlexGen (PyTorch / CUDA), Hugging Face Accelerate, and DeepSpeed ZeRO-Inference. Evaluated across OPT (6.7B, 13B, 30B, 66B), LLaMA (7B, 13B, 30B), and GPT-NeoX-20B.

---

## 3. How the Algorithm Actually Works (Under the Hood)

- **Data Structures Added / Modified:**
  - **Cumulative Attention Score Vector ($\tilde{o}_i \in \mathbb{R}^n$):** A running 1D tensor per head tracking the sum of attention probabilities assigned to each past token from generation step $1$ to $i$.
  - **Dual-Region Cache Buffer ($S_i$):** A pre-allocated tensor buffer of capacity $k = 2K$ per head:
    - *Heavy-Hitter Pool:* $K$ entries storing Key and Value vectors for tokens with the highest accumulated attention scores.
    - *Recent Window Pool:* $K$ entries maintained as a circular queue holding the most recent $K$ generated tokens.
  - **Eviction Tracking Tensor (`evict_ids`):** Integer index tensor passed to low-level CUDA kernels to overwrite evicted memory positions in-place without memory reallocation.

- **Scoring, Compression & Decision Logic:**
  1. *Normalized Attention Score Computation at Step $i$:*
     $$D_i = \left(\exp\left(Q_{i,*} (K_{S_{i-1},*})^\top\right) - \mathbf{1}_{[i]\setminus S_{i-1}}\right) \mathbf{1}_i$$
     $$o_i = D_i^{-1} \cdot \left(\exp\left(Q_{i,*} (K_{S_{i-1},*})^\top\right) - \mathbf{1}_{[i]\setminus S_{i-1}}\right)$$
     *Plain English:* $Q_{i,*}$ is the query vector of the newly generated token, and $K_{S_{i-1},*}$ represents the key vectors currently stored in the cache. $D_i$ is the normalization scalar (the softmax denominator) adjusted so that non-cached tokens contribute zero, and $o_i$ is the resulting normalized attention score vector across the retained tokens.
  2. *Cumulative Attention Score Accumulation:*
     $$\tilde{o}_i = \tilde{o}_{i-1} + o_i$$
     *Plain English:* The new attention probabilities $o_i$ are added element-wise into the cumulative score vector $\tilde{o}_{i-1}$. Tokens that receive repeated attention across multiple generation steps steadily grow large cumulative scores.
  3. *Greedy Eviction Update Rule:*
     $$F_{\text{score}}(T) = \sum_{s \in T} \tilde{o}_{i, s}$$
     $$u = \arg\max_{v \in (S_{i-1} \cup \{i\})} F_{\text{score}}\left((S_{i-1} \cup \{i\}) \setminus \{v\}\right) = \arg\min_{v \in (S_{i-1} \cup \{i\})} \tilde{o}_{i, v}$$
     $$S_i = (S_{i-1} \cup \{i\}) \setminus \{u\}$$
     *Plain English:* To choose which token to evict when the cache is full, evaluate all candidates in the current cache plus the new token. The token $u$ with the lowest accumulated attention score is evicted, ensuring the retained set $S_i$ maximizes total attention coverage.
  4. *Theoretical Submodular Approximation Guarantee (Theorem 4.4 & Theorem D.32):*
     $$f(\tilde{S}_i) \ge (1 - 1/e)(1 - \alpha) \text{opt}_i - \beta$$
     *Plain English:* Assuming the attention score coverage function is monotone submodular, this online greedy eviction policy is mathematically guaranteed to achieve at least $(1 - 1/e) \approx 63.2\%$ of the theoretical optimal offline cache value, within controllable error bounds $\alpha$ and $\beta$.
  - **Execution Cadence:** Prefill computes initial heavy hitters if the prompt length exceeds $2K$. During decoding, score accumulation and eviction execute at **every single autoregressive token step**.

- **Memory Hierarchy & Placement:**
  - **Keys ($K$) & Values ($V$):** Stored in FP16 / BF16 (or INT4 when evaluating quantization compatibility).
  - **Placement:** Active retained KV entries live exclusively in GPU VRAM. Evicted entries are permanently discarded.
  - **Granularity:** Token-level within each attention head and layer. Heavy-hitter selection operates independently per head, allowing different heads to track different critical tokens.

- **Decode-Phase Overheads:**
  - Finding the lowest-scoring token among $2K$ entries takes negligible time ($O(K)$ minimum search per head, taking $<10\,\mu\text{s}$ on GPU).
  - Zero memory reallocation overhead due to static circular buffer pre-allocation.
  - Stalls occur primarily if the inference engine must materialize full attention score vectors to GPU global memory instead of fusing attention inside on-chip SRAM.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

- **Memory Saved (Efficiency):**
  - **Cache Budget Reduction:** Operates with a KV cache budget ranging from **4% to 20%** of full sequence length.
  - **Compression Ratio:** Delivers a **$5\times\text{--}10\times$** memory footprint reduction for the KV cache while maintaining baseline performance.
  - **VRAM Footprint Impact:** For OPT-30B with sequence length 1,024 and batch size 128, slashes KV cache memory from 180 GB down to ~36 GB (at a 20% budget).

- **Speed & Latency:**
  - **Throughput Improvements on NVIDIA T4 GPU (16 GB VRAM):**
    - *Synthetic Benchmark (Prompt 512, Generation 512, OPT-6.7B):* H2O achieves **51.7 tokens/s** (batch size 4, pure GPU) vs FlexGen's 16.8 tokens/s (batch size 1, GPU), Accelerate's 15.5 tokens/s, and DeepSpeed's 9.6 tokens/s ($3.1\times$ over FlexGen, $3.3\times$ over Accelerate, $5.4\times$ over DeepSpeed).
    - *Synthetic Benchmark (Prompt 512, Generation 1024, OPT-30B, CPU Offload):* H2O achieves **13.82 tokens/s** (batch size 264) vs FlexGen's 7.1 tokens/s (batch size 48), Accelerate's 0.6 tokens/s, DeepSpeed's 0.6 tokens/s ($1.95\times$ over FlexGen, $23.0\times$ over Accelerate and DeepSpeed).
    - *Real-World Tasks (XSUM, OPT-6.7B):* H2O achieves **30.40 tokens/s** (batch size 1, GPU) vs FlexGen's 10.80 tokens/s ($2.81\times$), Accelerate's 11.98 tokens/s ($2.54\times$), DeepSpeed's 3.52 tokens/s ($8.64\times$).
    - *Real-World Tasks (XSUM, OPT-30B):* H2O achieves **6.70 tokens/s** (batch size 180, CPU offload) vs FlexGen's 3.29 tokens/s (batch size 44, $2.04\times$), DeepSpeed's 0.31 tokens/s ($21.6\times$), Accelerate's 0.23 tokens/s ($29.1\times$).
  - **End-to-End Latency on NVIDIA A100 GPU (80 GB VRAM):**
    - *Prompt 7000 + Generation 1024 (OPT-30B, batch size 1):* FlexGen latency $57.0\text{ s}$ $\to$ H2O latency **$50.4\text{ s}$** ($1.13\times$ speedup).
    - *Prompt 5000 + Generation 5000 (OPT-13B, batch size 4):* FlexGen latency $214.2\text{ s}$ $\to$ H2O latency **$155.4\text{ s}$** ($1.38\times$ speedup).
    - *Prompt 2048 + Generation 2048 (OPT-6.7B, batch size 24):* FlexGen latency $99.5\text{ s}$ $\to$ H2O latency **$53.5\text{ s}$** ($1.86\times$ speedup); throughput increases from 494.1 tokens/s to **918.9 tokens/s** ($1.86\times$), and scaling batch size to 64 yields **1161.0 tokens/s** (where FlexGen encounters Out-Of-Memory).
  - **Tail Latency (P95/P99):** Not reported in paper (mean throughput and total wall-clock generation latency reported only).

- **Output Quality & Trustworthiness:**
  - **5-shot Downstream Accuracy (OPT-30B with 20% KV cache budget):**
    - *PiQA:* Full Cache 78.51% $\to$ H2O **78.45%** (Local-only w/o H2 drops to 55.82%).
    - *COPA:* Full Cache 85.00% $\to$ H2O **84.00%** (Local-only w/o H2 drops to 48.00%).
    - *OpenBookQA:* Full Cache 43.20% $\to$ H2O **43.00%** (Local-only w/o H2 drops to 25.20%).
    - *Winogrande:* Full Cache 70.24% $\to$ H2O **69.06%** (Local-only w/o H2 drops to 49.17%).
  - **Component Ablation on OPT-30B (Table 9):**
    - *MathQA:* Full 26.23% vs Local-only 20.87% vs H2-only 21.98% vs Combined (H2O) **26.87%** (surpassing full cache by 0.64%).
  - **Extreme Context Streaming (Figure 5 & Appendix C.4):**
    - Tested on PG-19 with inputs up to **4,000,000 tokens** using streaming positional rolling: H2O maintains stable perplexity (~15.2--15.8) on par with full-cache models, whereas a sliding window without heavy hitters experiences catastrophic perplexity explosion ($>10^3$).

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Fully compliant. Model weights remain 100% untouched in native BF16 precision.
  - **Dense GQA Compatibility:** The original paper evaluated Multi-Head Attention (MHA) models (OPT, LLaMA-1, GPT-NeoX). Our project's pinned models (**Qwen 3 4B**, **Phi-4 mini**, **Ministral 3 3B**, **SmolLM3 3B**, **Qwen 3 1.7B**) all use **Grouped-Query Attention (GQA)** (4:1 or 8:1 query heads per KV head). In GQA, multiple query heads share a single physical KV head. To apply H2O to GQA models, cumulative attention scores must be pooled (averaged or summed) across all query heads belonging to the same KV head group before computing eviction ranks.

- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - On our local 8 GB RTX 4060, a 4B parameter model in BF16 occupies ~8 GB of VRAM, leaving virtually zero space for KV cache. Capping the KV cache to a 20% budget allows processing multi-thousand token contexts without immediate Out-Of-Memory crashes.
  - On our 16 GB RTX 4090 cluster cap, a 20% KV cache budget allows increasing concurrent batch sizes from 2 up to 8–12 requests for 8K context lengths, directly supporting our 10–15 concurrent SME user target.

- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** Medium for standalone PyTorch/FlexGen benchmarking; High for native vLLM PagedAttention integration.
  - **WP-3 Micro-Benchmarking:** Standalone attention hooks can easily extract cumulative attention vectors to benchmark scoring accuracy, sparsity ratios, and GQA head pooling on our pinned models.
  - **WP-6 Engine Integration (`vLLM` PagedAttention / `LMCache`):**
    - *The Virtual Paging Block-Fragmentation Challenge:* H2O selects individual tokens for eviction. In vLLM, memory is allocated in contiguous physical blocks (typically 16 or 32 tokens). Evicting arbitrary single tokens fragments block tables, disables contiguous tensor operations, and degrades FlashAttention efficiency.
    - *Adaptation for Our Engine:* Rather than single-token eviction, we can adapt H2O scoring to **block-level importance pooling** (summing cumulative attention scores across all tokens in a 16-token page). Most crucially, instead of permanently deleting cold blocks, our engine uses H2O's cumulative score as the **priority ranking signal for WP-4**: hot heavy-hitter blocks remain in GPU VRAM in BF16, while lower-priority blocks are quantized to INT4 or offloaded to host CPU memory via LMCache.

- **Classification for Project:** **Core Target for Reproduction & Component Inspiration** (Serves as the foundational attention-importance scoring mechanism for structural eviction and tiered offloading in WP-4 and WP-6).
