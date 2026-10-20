# EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  In multi-turn, long-running chats (spanning days or weeks), the KV cache quickly explodes (for example, a 3B model's KV cache exceeds 7 GB after just 30 sessions, which is bigger than the model itself). Most existing methods have two flaws:
  1. *Unbounded Peak Memory:* They wait until the entire massive prompt is processed before throwing away tokens ("post-prefill eviction"), which means you still crash with Out-Of-Memory during the initial prompt reading.
  2. *Multi-Turn Amnesia:* If you evict tokens based only on the current user question, you permanently delete facts mentioned early on that become critical 10 turns later when the topic switches.

- **How EpiCache Solves It (Step-by-Step in Plain Words):**
  1. **Chop & Group by Topic (Offline Clustering):** It slices historical dialogue into short 4-utterance chunks and uses a small embedding model with K-Means clustering to group them into coherent topics (called "episodes"—like *travel*, *work*, *movies*). It finds the single most central, representative dialogue chunk for each topic.
  2. **Compress Block-by-Block (During Prefill):** Instead of processing the whole history at once, it feeds the chat into the model in small blocks (e.g., 512 tokens). After each block, it immediately discards less important tokens so GPU memory stays strictly capped. To decide what to keep for a given topic, it temporarily attaches that topic's representative chunk as a "fake question" (patched prompt) to see which tokens the model actually attends to.
  3. **Tiered Topic Caches in Host RAM (Storage):** This generates a small, compressed KV cache for each topic (e.g., 4 topics total). Inactive topic caches sit safely in host CPU RAM, so active GPU memory remains tiny.
  4. **Dynamic Topic Routing (During Decode):** When a new user question arrives, EpiCache embeds the question, matches it to the closest topic centroid, pulls that topic's cache into GPU VRAM, and generates the answer. If the topic changes later, it writes back the updated cache to CPU RAM and fetches the new one.

- **How It Compares to Our Project:**
  Like our 3-axis project, EpiCache combines **Structural** (token eviction and layer budgeting) and **Spatial** (offloading inactive caches to CPU RAM). However, EpiCache *permanently deletes* tokens that fall outside the topic's top-$M$ budget. In contrast, our project aims to preserve long-term context by *compressing and quantizing* (e.g., to 4-bit) the lower-priority tokens and offloading them rather than permanently evicting them.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Peak GPU memory is slashed by up to $3.7\times$ (staying flat around $9.6\text{ GB}$ on a $90\text{K}$ context, avoiding OOM on consumer GPUs); decoding is up to $2.4\times$ faster ($18\times$ faster than re-reading the prompt from scratch); beats RAG and standard eviction by up to $30\%$ accuracy on long conversation benchmarks.
  - **What You Pay:** Offline prefill must be run once for each topic ($E \times$ prefill passes); requires keeping a small $0.6\text{B}$ embedding model around; incurs a ~0.3-second PCIe transfer stall whenever the user abruptly shifts topics.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Layer Sensitivity is Model-Dependent, Not Data-Dependent:** The authors discovered that certain transformer layers are extremely sensitive to having their KV tokens evicted, while other layers barely notice. Crucially, this sensitivity pattern is fixed for a given model architecture and stays identical across code, books, and conversation! We can profile our pinned models (`Qwen3-4B`, `Phi-4 mini`) *once offline using a single calibration sample* and assign larger VRAM budgets to sensitive layers.
  2. **Block-Wise Prefill Caps Peak VRAM:** Pruning the KV cache block-by-block during prompt processing guarantees that peak GPU memory never exceeds the budget plus one block ($M + M_{\text{block}}$). This is vital for fitting 32K context on 8–16 GB GPUs.
  3. **Topic Locality Prevents PCIe Thrashing:** In real user conversations (tested on RealTalk), users stay within the same topic ~70% of the time. This means the CPU-to-GPU cache swap only triggers on ~30% of turns, avoiding constant PCIe transfer overhead.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Evaluated on High-End Hardware (NVIDIA A100):** The paper measures swap latency on an A100. Over standard consumer PCIe (Gen 4 x8 or x16 on an RTX 4060 / local rig), swapping uncompressed FP16 topic caches will take longer and could cause noticeable tail-latency pauses if the user frequently hops topics.
  - **Token Eviction is Permanent:** While topic clustering helps retain topic-specific facts, any fine-grained detail discarded during the block eviction step is gone forever. If a question requires cross-referencing an evicted detail, the model fails.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our manual notes correctly identified the core idea: dissecting conversations into blocks, grouping by topic (episodes), keeping topic centroids via K-Means, and warning that A100 testing might mask consumer hardware bottlenecks.
  - *Clarification from PDF:* The manual note stated it "dissects whole conversation into block, group block by topic". To be precise: it actually clusters *utterance segments* (4 turns) using a separate text embedding model first, and then performs *block-wise prefill eviction* over the entire text using the cluster's medoid as a patched scoring prompt.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Minsoo Kim, Arnav Kundu, Han-Byul Kim, Richa Dixit, Minsik Cho (Apple)
- **Publication / Venue & Year:** Proceedings of the 43rd International Conference on Machine Learning (ICML 2026), PMLR 306, 2026. (arXiv:2509.17396v4, May 2026)
- **Primary Axis:** Structural (Episodic attention-guided block-wise token eviction & sensitivity-aware layer budgeting)
- **Secondary Interacting Axis:** Spatial (Multi-cache episodic tiering in host memory with query-time routing to GPU)
- **Official Codebase / Repository:** Apple / OpenReview (PMLR 306)
- **Baseline Implementation Stack:** HuggingFace Transformers / PyTorch custom block-prefill harness, evaluated on LLaMA-3.2-3B, LLaMA-3.1-8B, Qwen2.5-3B, Qwen2.5-7B, Qwen3-4B, and Qwen3-8B.

---

## 3. How the Algorithm Actually Works (Under the Hood)

- **Data Structures Added / Modified:**
  - **Centroids & Medoids:** $E$ topic centroid embedding vectors ($c_e \in \mathbb{R}^d$) and the actual text segments closest to each centroid ($S_{\text{medoid}}^{(e)}$).
  - **Episodic Cache Bank:** A collection $\mathcal{B} = \{\mathcal{C}_{\text{KV}}^{(1)}, \dots, \mathcal{C}_{\text{KV}}^{(E)}\}$ holding $E$ independent KV caches.
  - **Layer Sensitivity Vector:** A static per-layer sensitivity vector $s_\ell = 1 - \sigma_\ell \in \mathbb{R}^L$ that determines how much cache budget each layer gets.

- **Scoring, Compression & Decision Logic:**
  1. *Patched Cross-Attention Scoring:* At each block boundary of $M_{\text{block}}$ tokens, the medoid segment $P_e$ of length $p$ is appended as an auxiliary prompt. Token importance $s_i$ is scored by maximum attention received from the prompt tokens:
     $$s_i^{\max} = \max_{t \in [n+1, n+p]} \text{Attn}(x_t \to x_i)$$
     *Plain English:* For every past token $x_i$, look at how much attention each token $x_t$ in the representative prompt paid to it, and keep the highest attention score as its importance.
  2. *Layer Sensitivity Metric:* For layer $\ell$, Key states are generated under a full causal mask $\mathcal{M}$ and under a restricted budget mask $\mathcal{M}'$. The sensitivity $\sigma_\ell$ is the average cosine similarity between these Key vectors:
     $$\sigma_\ell = \frac{1}{H N} \sum_{h=1}^H \sum_{i=1}^N \cos\left(k_{\text{full}, i}^{(\ell, h)}, k_{\text{block}, i}^{(\ell, h)}\right), \quad s_\ell = 1 - \sigma_\ell$$
     *Plain English:* Compare the Key vectors produced with full context against Key vectors produced under memory limits. If the similarity is high ($\sigma_\ell \approx 1$), the layer doesn't care much about missing context ($s_\ell \approx 0$). If similarity drops, the layer is sensitive ($s_\ell$ is large).
  3. *Sensitivity-Aware Budget Allocation:* Distribute the global token budget $(L \cdot M)$ across layers proportionally:
     $$M_\ell^{\text{alloc}} = \frac{s_\ell^\alpha}{\sum_{j=1}^L s_j^\alpha} \cdot (L \cdot M)$$
     *Plain English:* Give more cache slots to sensitive layers and fewer slots to insensitive layers. $\alpha$ controls how aggressively we favor sensitive layers (empirically best at $\alpha = 2\text{--}4$).
  - **Execution Cadence:** Clustering and layer sensitivity profiling run once offline. Block-prefill eviction runs during initial ingestion. Query routing runs once per conversational turn.

- **Memory Hierarchy & Placement:**
  - **Keys & Values:** Stored in BF16/FP16 (no quantization used).
  - **Placement:** Only 1 active episodic cache ($\mathcal{C}_{\text{KV}}^{(e^\dagger)}$) of budget $M$ sits in GPU VRAM during decoding. The remaining $E-1$ episodic caches reside in host CPU memory.
  - **Granularity:** Token-level eviction within blocks; layer-level budget allocation; episode-level swapping.

- **Decode-Phase Overheads:**
  - Query embedding and cosine matching takes $\approx 0.04\text{ s}$.
  - On a topic shift (~30% of turns), swapping the episode cache over PCIe takes $\approx 0.30\text{ s}$ on an A100.
  - On continuous turns (~70% of turns), the cache remains resident on GPU with $0\text{ s}$ transfer delay.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

- **Memory Saved (Efficiency):**
  - **Peak VRAM on LLaMA-3.2-3B ($90\text{K}$ context):** Dropped from $36.3\text{ GB}$ (Full KV) down to $9.6\text{ GB}$ ($3.78\times$ reduction), staying completely flat across sequence length.
  - **Cache Budgets Evaluated:** Evaluated under $M \in \{2\text{K}, 4\text{K}, 6\text{K}, 8\text{K}\}$ tokens across contexts up to $100\text{K}$ tokens.

- **Speed & Latency:**
  - **Total Session Latency (300 turns over 90K history):**
    - Naive Full KV (re-prefilling every turn): $9,339\text{ s}$ ($31.1\text{ s/turn}$).
    - Full KV with Prefix Caching: $1,062.8\text{ s}$ ($3.5\text{ s/turn}$, dominated by decoding over 90K tokens).
    - EpiCache ($M=8\text{K}$): $545.4\text{ s}$ ($1.8\text{ s/turn}$, $1.4\text{ s}$ decode + amortized swaps), achieving an overall $1.95\times$ speedup over prefix caching and $17.1\times$ speedup over full re-prefill.
  - **Tail Latency (P95/P99):** Not reported as a distribution. The per-swap latency is reported as $0.30\text{ s}$ when a topic shift occurs.

- **Output Quality & Trustworthiness:**
  - **LongMemEval ($21.8\text{K}$ history, LLaMA-3.2-3B):**
    - Full KV: $39.4\%$ accuracy.
    - EpiCache ($M=8\text{K}$): $39.5\%$ accuracy (100% quality retention).
    - Baselines at $M=8\text{K}$: KVzip ($37.5\%$), InfiniPot ($36.2\%$), SnapKV ($35.6\%$).
    - Extreme Compression ($M=2\text{K}$): EpiCache achieves $34.4\%$ vs. KVzip ($18.2\%$) and InfiniPot ($26.3\%$).
  - **RealTalk ($26.4\text{K}$ history, LLaMA-3.1-8B):**
    - At $M=8\text{K}$, EpiCache achieves $47.5\%$ average score vs. KVzip ($40.3\%$) and RAG-2K ($31.9\%$).
    - On cross-episode questions, EpiCache reaches $50.8\%$ vs. RAG's $31.9\%$.

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Fully compliant. Model weights remain untouched in native BF16 precision.
  - **Dense GQA Compatibility:** Evaluated directly on our primary benchmark models (**Qwen3-4B**, **Qwen2.5-3B**, **LLaMA-3.1-8B**). Attention scores in Equation (3) are pooled across grouped query heads.
- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - High feasibility for single-user workloads: An $8\text{K}$ cache for Qwen3-4B takes only $\approx 1.12\text{ GB}$ of VRAM. Combined with $8\text{ GB}$ model weights, this operates well within our $16\text{ GB}$ VRAM cap.
  - Host RAM requirement: $E=4$ episodic caches occupy $\approx 3.3\text{ GB}$ of host memory, fitting into our $16\text{ GB}$ system RAM specification.
- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** Medium. Clustering is standard scikit-learn / PyTorch. Block prefill can be implemented via attention masking hooks.
  - **WP-3 Micro-Benchmarking:** Excellent candidate to benchmark layer sensitivity profiling and block-prefill eviction latency against baseline vLLM.
  - **WP-6 Engine Integration (`vLLM` PagedAttention / `LMCache`):**
    - *Challenge:* EpiCache performs fine-grained token-level eviction during prefill, which fragments virtual block tables in vLLM PagedAttention.
    - *Adaptation for Our Engine:* Rather than arbitrary token eviction, we can apply eviction at **block / page boundaries** (e.g., 16-token chunks) or use LMCache's chunk-level tiered storage to move entire inactive episode blocks between GPU and CPU.
- **Classification for Project:** **Core Target for Reproduction & Component Inspiration** (Provides a concrete foundation for multi-turn conversational session management in WP-4).
