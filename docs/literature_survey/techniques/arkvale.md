# ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  Existing KV cache pruning methods assume that once an older token is no longer attended to, it is useless forever and can be permanently deleted. However, in real conversations, document reading, and reasoning, token importance shifts dynamically over time. A detail mentioned thousands of tokens ago might be ignored for twenty paragraphs, but then suddenly become essential when a specific question asks about it. When existing systems permanently throw away those dormant tokens, they suffer severe amnesia and accuracy collapse (often failing long-context retrieval tasks more than sixty percent of the time).

- **How It Solves It (Step-by-Step in Plain Words):**
  1. **Before / During Prefill (Grouping and Summarizing):** Instead of managing individual tokens one by one, ArkVale groups tokens into fixed-size pages (such as blocks of sixteen or thirty-two tokens, matching the memory pages used by operating systems). As soon as a page is filled with tokens, ArkVale creates a tiny geometric summary of the key vectors in that page (called a page digest). This digest is a compact bounding box that records the upper and lower coordinate boundaries of the keys in that page.
  2. **Where the Data Lives (Storage & Tiering):** The full, uncompressed keys and values of every completed page are copied asynchronously over the computer bus to host CPU memory as an emergency backup. Only a small, fixed budget of active pages (for example, two thousand to four thousand tokens) is kept in fast GPU video memory. Crucially, the tiny page digests for *every single page*—both those in GPU memory and those backed up on the CPU—remain stored on the GPU.
  3. **During Decode (Generating New Tokens):** Before computing attention for each newly generated token, the GPU uses the new query vector and the tiny page digests to rapidly estimate an upper bound on the attention score each page could possibly receive. It ranks all pages across the entire prompt history. If an evicted page sitting in CPU memory lands in the top ranks, ArkVale immediately pulls it back into GPU memory across the PCIe bus (a recall operation), while dropping the lowest-ranked page currently in GPU memory. Because consecutive decoding steps share almost the same set of important pages, only a tiny number of pages (typically fewer than ten pages, or less than five megabytes) need to be swapped per token, preventing communication stalls.

- **How It Compares to Our Project:**
  ArkVale combines **Structural** (page-level token eviction) and **Spatial** (CPU offloading and on-demand recall) optimization. Unlike permanent eviction baselines, it maintains a safety net in CPU RAM. However, ArkVale stores the entire CPU backup in uncompressed sixteen-bit floating point, whereas our project goes a step further by pairing priority scoring with selective precision quantization (e.g., four-bit) for offloaded or lower-priority tokens to fit larger batches within limited host RAM.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Caps active GPU KV cache memory to a strict two-thousand to four-thousand token budget; speeds up long-context decoding latency by up to 2.2 times (1.7 times on average) and boosts serving throughput by up to 4.6 times; maintains ninety-five to one hundred percent retrieval accuracy on needle-in-a-haystack tasks where standard eviction baselines plummet below twenty percent.
  - **What You Pay:** Requires sufficient host CPU memory to store full uncompressed backups of all processed tokens; adds a small GPU computation overhead at every single step to calculate bounding-box dot products across all digests; backing up pages during the initial prompt reading phase cannot be completely hidden asynchronously.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Bounding-Box Digests for Lightning-Fast Screening:** Summarizing a block of keys by its axis-aligned bounding box (the minimum and maximum coordinate boundaries) allows the GPU to compute a mathematical upper bound on the attention score using simple element-wise operations. This is vastly cheaper than storing all keys or computing full attention, yet accurate enough to screen out irrelevant pages without ever fetching them from CPU memory.
  2. **Zero-Copy Eviction via Proactive Backup:** Because every page is copied to host RAM immediately when filled, evicting a page from GPU memory during decoding has zero write overhead—the GPU simply reassigns or overwrites that memory buffer without sending anything back across the PCIe bus. Only recalled pages consume PCIe transfer bandwidth.
  3. **Natural Alignment with PagedAttention Blocks:** ArkVale operates at the granularity of pages (sixteen or thirty-two tokens), which matches the exact physical block size used by modern serving engines like vLLM and LMCache. This completely avoids the severe memory fragmentation caused by fine-grained, token-level eviction algorithms.
  4. **Exempting Early Attention Layers:** The authors observed that the first two transformer layers exhibit almost no token sparsity (all tokens receive non-trivial attention). By leaving the first two layers unevicted at full capacity, the model preserves foundational syntax representations while concentrating pruning on deeper, highly sparse layers.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Host RAM Capacity Bottleneck:** The authors tested on a high-end server with an Intel Xeon CPU and eighty gigabytes of GPU memory where host memory was abundant. On a consumer machine with only sixteen gigabytes of system RAM, storing uncompressed sixteen-bit backups for long contexts across multiple concurrent users will quickly trigger operating system swapping or out-of-memory errors unless combined with quantization.
  - **PCIe Latency on Gen 4 x8:** On high-end servers with full sixteen-lane PCIe connections, transferring five megabytes takes under two hundred microseconds. On a consumer RTX 4060 laptop or desktop running on an eight-lane connection (roughly fourteen gigabytes per second effective), any sudden burst in recalled pages could start eating into the decoding time budget if the cache capacity is set too small.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our team skimming notes correctly highlighted the core mechanics: grouping tokens into pages, keeping a full CPU backup with a tiny GPU summary, using a geometric summary (bounding box) rather than a learned score, and recalling top-ranked pages step-by-step.
  - *Clarification from PDF:* The team note questioned whether "scoring cost a lot" for the bounding-box digest. The PDF proves that scoring all digests is computationally negligible because it reduces to an element-wise vector multiply-add on two vectors per page, which runs in microseconds on the GPU. Furthermore, the PDF reveals that an Axis-Aligned Bounding Box with a "mean" radius vector (`cuboid-mean`) achieves ninety-five percent top-one recall accuracy, dramatically outperforming a naive key centroid dot-product baseline (which achieves less than five percent top-one accuracy).

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Renze Chen, Zhuofeng Wang, Beiquan Cao, Tong Wu, Size Zheng, Xiuhong Li, Xuechao Wei, Shengen Yan, Meng Li, Yun Liang (Corresponding Author) — Peking University & Infinigence-AI
- **Publication / Venue & Year:** 38th Conference on Neural Information Processing Systems (NeurIPS 2024). OpenReview Forum: `4oAt5L4lYe`.
- **Primary Axis:** Structural (Page-level KV cache eviction & dynamic bounding-volume attention selection)
- **Secondary Interacting Axis:** Spatial (Asynchronous CPU host backup and PCIe-based dynamic page recall)
- **Official Codebase / Repository:** [https://github.com/pku-liang/ArkVale](https://github.com/pku-liang/ArkVale)
- **Baseline Implementation Stack:** Built on top of HuggingFace Transformers (v4.40.0), PyTorch (v2.3.0), and CUDA 12.3, using CUTLASS, FlashInfer, and RAFT for specialized GPU kernels. Evaluated on `LongChat-7b-v1.5-32k` (LLaMA-2 architecture).

---

## 3. How the Algorithm Actually Works (Under the Hood)

- **Data Structures Added / Modified:**
  - **Paged KV Buffers:** Token KV pairs are partitioned into non-contiguous physical pages of size $p \in \{16, 32\}$ tokens.
  - **GPU Page Digest Table:** For each allocated page $j$, stores two compact boundary vectors representing the Axis-Aligned Bounding Box (AABB) of key vectors: $b_j^{\max}, b_j^{\min} \in \mathbb{R}^d$ (or center $c_j \in \mathbb{R}^d$ and scalar radius $r_j \in \mathbb{R}$ for bounding spheres).
  - **Host Memory Backup Bank:** Pinned CPU host memory pool holding full BF16/FP16 copies of keys and values for all filled pages.
  - **Page Residency Tracker:** Bitmask tracking whether each page resides in GPU VRAM or is currently evicted to CPU RAM.

- **Scoring, Compression & Decision Logic:**
  1. *Bounding-Sphere Digest Formulation:*
     The keys in a page $K = \{k^{(i)}\}_{i=1}^p$ with $k^{(i)} \in \mathbb{R}^d$ can be enclosed in a hypersphere with center $c \in \mathbb{R}^d$ and radius $r \in \mathbb{R}$:
     $$I_{\text{sphere}}(q, K) = \max_{k \in K} q \cdot k \approx q \cdot \left(c + \frac{r}{\|q\|_2} q\right) = q \cdot c + r \|q\|_2$$
     *Plain English:* To find the highest possible dot product between query vector $q$ and any key inside the sphere, take the dot product with the sphere center $c$ and add the maximum distance along the query's direction, which equals the radius $r$ multiplied by the query vector's Euclidean norm $\|q\|_2$.
     The center is the midpoint of the page's coordinate bounds: $c = \frac{1}{2}\left(\min_{k \in K} k + \max_{k \in K} k\right)$. The radius alternatives are:
     $$r_{\max} = \max_{k \in K} \|c - k\|_2, \quad r_{\text{center}} = \frac{1}{2}\left(\min_{k \in K} \|c - k\|_2 + r_{\max}\right), \quad r_{\text{mean}} = \frac{1}{|K|}\sum_{k \in K} \|c - k\|_2$$
     *Plain English:* $r_{\max}$ strictly encloses all keys, while $r_{\text{center}}$ and $r_{\text{mean}}$ tighten the boundary to prevent overestimating page importance during decode ranking.

  2. *Bounding-Cuboid (AABB) Digest Formulation (Default & Best Performing):*
     The page keys are enclosed by an Axis-Aligned Bounding Box with lower corner $b^{\min} \in \mathbb{R}^d$ and upper corner $b^{\max} \in \mathbb{R}^d$:
     $$I_{\text{cuboid}}(q, K) = \max_{k \in K} q \cdot k \approx \sum_{i=1}^d \max_{k \in K}(q_i k_i) = \sum_{i=1}^d \max(q_i b_i^{\max}, q_i b_i^{\min}) = \mathbf{1}^\top \max(q \odot b^{\max}, q \odot b^{\min})$$
     *Plain English:* For each hidden dimension $i$, the maximum product between query scalar $q_i$ and any key coordinate must occur at either the upper bound $b_i^{\max}$ (if $q_i > 0$) or the lower bound $b_i^{\min}$ (if $q_i < 0$). Summing these maximums across all hidden dimensions gives a tight upper bound on the attention logit for that entire page.
     The cuboid boundaries are computed using coordinate radius vector $r \in \mathbb{R}^d$:
     $$r_{\text{mean}} = \frac{1}{|K|}\sum_{k \in K} |c - k|, \quad b^{\max} = c + r_{\text{mean}}, \quad b^{\min} = c - r_{\text{mean}}$$
     *Plain English:* For each coordinate, compute the average absolute distance from the center $c$. Setting the box limits to $c \pm r_{\text{mean}}$ produces the balanced bounding box that empirically yielded 95% top-1 recall accuracy.

  3. *Page Ranking & Attention Selection:*
     At each decode step, all pages are ranked by estimated score $I(q, K)$. The system selects the top-$k$ pages to participate in attention, where:
     $$k = \frac{\min(C, c/2)}{p}$$
     *Plain English:* Select $k$ pages such that the total tokens participating in attention never exceed a threshold $C$ (default 1280 tokens) and never exceed half the total GPU cache capacity $c$. Any selected page currently in CPU memory is recalled; unselected GPU pages are dropped to maintain cache capacity.
  - **Execution Cadence:** Bounding-box digest creation runs once when each page of $p$ tokens is filled. Page scoring, re-ranking, and selective recall execute at every single decoding step. Layers 0 and 1 are permanently retained without eviction.

- **Memory Hierarchy & Placement:**
  - **Keys ($K$) & Values ($V$):** Active top-$k$ pages reside in GPU VRAM in uncompressed FP16/BF16. Full backup of all generated pages resides in host CPU RAM in FP16/BF16.
  - **Page Digests ($b^{\max}, b^{\min}$):** Stored permanently in GPU VRAM in FP16 ($2 \times d$ values per page of $p$ tokens).
  - **Granularity:** Page-level (blocks of 16 or 32 tokens).

- **Decode-Phase Overheads:**
  - Scoring all digests takes $< 0.1\text{ ms}$ on GPU via fused CUDA/Triton kernels.
  - Average pages recalled per step: $< 10$ pages ($\approx 1.3\text{--}5.0\text{ MB}$ over PCIe).
  - PCIe transfer latency: $100\text{--}300\ \mu\text{s}$, accounting for only a small fraction of the total decoding step (~15–50 ms).

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

- **Memory Saved (Efficiency):**
  - **GPU VRAM Footprint:** Caps active GPU KV cache to 2,048–4,096 tokens regardless of context length (e.g., maintaining flat KV memory usage across 32K context).
  - **Throughput Scaling:** Under a 40 GB KV memory cap on NVIDIA A100, ArkVale achieves up to **$4.6\times$ higher serving throughput** ($3.5\times$ on average across sequence lengths 10K, 20K, and 30K) by enabling significantly larger batch sizes.

- **Speed & Latency (TTFT, TPOT, Tail Latency):**
  - **Decoding Latency (GovReport, batch-size=4, A100):**
    - *At 10K Context:* Full KV baseline $\approx 15\text{ ms}$ vs. ArkVale $\approx 8\text{--}10.5\text{ ms}$ (budget 512–4096).
    - *At 20K Context:* Full KV baseline $\approx 28\text{ ms}$ vs. ArkVale $\approx 14\text{--}18\text{ ms}$.
    - *At 30K Context:* Full KV baseline $\approx 43\text{ ms}$ vs. ArkVale $\approx 20\text{--}26\text{ ms}$ (up to **$2.2\times$ decoding speedup**, $1.7\times$ on average).
  - **Latency Breakdown per Token (Figure 9a):** Attention computation dominates ($> 70\%$ of time); Estimate takes $\approx 5\text{--}15\%$; Recall takes $< 5\%$; Select takes $< 2\%$.
  - **Tail Latency (P95/P99):** Not reported in paper (only mean latency breakdown and relative throughput are reported).

- **Output Quality & Trustworthiness:**
  - **Passkey Retrieval Accuracy (Table 1, LongChat-7b-v1.5-32k across 10K, 20K, 30K contexts):**
    - *StreamingLLM:* 0%–40% (collapses to 0% at 512 budget across 10K–30K).
    - *H2O:* 0%–40% (collapses to 0%–5% at 512–1024 budgets).
    - *TOVA:* 5%–40% (drops to 5%–10% at tight budgets).
    - *ArkVale:* **95%–100% across all tested context lengths (10K, 20K, 30K) and all cache budgets (512, 1024, 2048, 4096)**.
  - **LongBench Benchmark Scores (Figure 8, 6 datasets, budgets 512–4096):**
    - Near-identical to Full KV ("Origin") at budgets $\ge 2048$ tokens.
    - Matches Origin at budget 1024 for HotpotQA and Qasper; matches Origin at budget 512 for TriviaQA; matches Origin at budget 2048 for NarrativeQA and PassageRetrieval.
    - Page size $p=16$ slightly outperforms $p=32$ under extremely tight budgets (e.g., 512) because finer page granularity enables selecting more diverse token clusters.

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Fully compliant. ArkVale is completely training-free and leaves model weights untouched in native BF16 precision.
  - **Dense GQA Compatibility:** ArkVale was demonstrated on MHA (`LongChat-7B`). For our pinned **Grouped-Query Attention (GQA)** models (`Qwen 3 4B`, `Phi-4 mini`, `Ministral 3 3B`, `SmolLM3 3B`, `Qwen 3 1.7B`), keys are shared across query head groups (e.g., 4 or 8 query heads per key head). Digest bounding boxes are constructed per key head, and query vectors can compute upper bounds per key head by max-pooling across the grouped query heads before ranking.
- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - **VRAM Budget:** On an RTX 4060 (8 GB VRAM), loading `Qwen 3 4B` in BF16 consumes $\approx 8\text{ GB}$, leaving very little headroom. Under ArkVale with a 2K token cache budget ($p=32$, $c=2048$), the active KV cache occupies only $\approx 0.28\text{ GB}$ of VRAM!
  - **PCIe Transfer Overhead:** Recalling 10 pages ($10 \times 32 = 320$ tokens) for `Qwen 3 4B` requires transferring:
    $$320 \text{ tokens} \times 2 \times (8 \text{ KV heads} \times 128 \text{ dim}) \times 2 \text{ bytes} \approx 1.31\text{ MB}$$
    Over consumer PCIe Gen 4 x8 (~14 GB/s effective on RTX 4060), $1.31\text{ MB}$ takes under **$100\ \mu\text{s}$ ($0.1\text{ ms}$)**, which is less than 0.5% of a typical 25 ms decode step.
  - **Host RAM Bottleneck:** The main challenge on our hardware is the 16 GB system RAM specification. A single 32K context in FP16 occupies $\approx 4.5\text{ GB}$ in CPU memory. Serving 10–15 concurrent SME users would demand $> 45\text{ GB}$ of host RAM, overflowing system memory. This strongly justifies our project's approach: **compressing/quantizing offloaded CPU pages to 4-bit** rather than storing raw 16-bit backups.
- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** **Medium**. ArkVale's page-based abstraction aligns directly with `vLLM` PagedAttention and `LMCache`.
  - **WP-3 Micro-Benchmarking:** Digest estimation kernels (`cuboid-mean`) and PCIe page recall can be micro-benchmarked in PyTorch/Triton against raw FlashAttention decoding.
  - **WP-6 Engine Integration (`vLLM` PagedAttention / `LMCache`):**
    - ArkVale's $p=16$ or $p=32$ tokens per page maps 1-to-1 onto vLLM's `block_size`!
    - Instead of evicting individual tokens (which fragments vLLM block tables), ArkVale swaps whole blocks, preserving contiguous virtual memory layouts.
    - LMCache already provides CPU-GPU block transfer channels; we can hook ArkVale's bounding-box ranker into LMCache's cache eviction/retrieval policy.
- **Classification for Project:** **Component Inspiration for WP-4 & WP-6** (The bounding-box key digest provides an optimal, lightweight screening mechanism for deciding which offloaded CPU blocks to recall into GPU VRAM).
