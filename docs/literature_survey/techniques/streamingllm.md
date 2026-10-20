# StreamingLLM: Efficient Streaming Language Models with Attention Sinks

## 0. Plain-English TL;DR (What It Does & Why We Care)

- **The Core Problem:**
  When using large language models for continuous, long-running streaming tasks (such as endless multi-turn chatbots or real-time document assistants), two major roadblocks occur:
  1. *Unbounded Memory Growth:* The key-value memory cache expands with every single generated word until the graphics card runs out of memory and crashes.
  2. *The Window Attention Collapse:* The intuitive fix is a sliding window—only keeping the most recent couple thousand words in memory and dropping older words. However, the moment you delete even the very first word from the cache, the model catastrophically breaks down and outputs pure gibberish (its perplexity explodes from 5 to over 5,000).
  3. *The Re-computation Slowness:* You could avoid the crash by recalculating the entire recent window from scratch for every single new word generated, but this is brutally slow (up to twenty times slower), making real-time streaming impossible.

- **How StreamingLLM Solves It (Step-by-Step in Plain Words):**
  1. **Discovering the Attention Sink Phenomenon:** The authors discovered why sliding windows fail: the mathematical normalization in self-attention forces all attention scores across the conversation to sum up to exactly one. Even when the model does not need information from past words, it cannot give them zero attention; it is mathematically forced to dump excess, unneeded attention somewhere. Because the very first words of a sequence are seen by every single subsequent word during training, the model naturally learns to use the very first words as a numerical dump site (an "attention sink"), regardless of what those words actually mean.
  2. **The 4-Token Anchor:** When a sliding window deletes those first tokens, the mathematical denominator of the attention calculation collapses, wrecking the entire probability distribution. StreamingLLM fixes this by permanently keeping just the first four tokens in the graphics memory as numerical anchors.
  3. **The Rolling Window:** Alongside these four permanent anchor tokens, StreamingLLM keeps a standard rolling sliding window of the most recent tokens (e.g., the last 1,000 or 2,000 tokens) in memory. As new words arrive, the oldest words in the sliding window are evicted, while the four anchor tokens remain untouched.
  4. **The Cache-Relative Position Trick:** When words are discarded from the middle, the gaps would normally confuse the model's positional tracking. StreamingLLM reassigns positions based on where tokens sit inside the rolling memory buffer rather than their original positions in the raw text, keeping positional differences within familiar pre-trained ranges.

- **How It Compares to Our Project:**
  StreamingLLM is a pure **Structural** optimization that achieves constant memory and infinite generation length by anchoring initial sink tokens and rolling a recency window. However, StreamingLLM has a major limitation: it *permanently discards* every token in the middle. If a user asks a question about a detail mentioned ten minutes ago that rolled out of the window, StreamingLLM has zero recall and fails completely. In our 3-axis project, instead of throwing away those middle tokens, we keep the four sink tokens and the recent sliding window pinned on the graphics card in full precision, while compressing (quantizing) and offloading the evicted middle tokens to host system memory so they can be recalled when needed.

- **The Bottom-Line Trade-off:**
  - **What You Gain:** Stable, infinite-length text generation with zero model retraining; completely flat graphics memory usage that never runs out; up to a 22-fold decoding speedup over sliding-window re-computation; handles over 4 million tokens without perplexity degradation.
  - **What You Pay:** Permanent amnesia for all evicted middle context (accuracy drops to zero on questions about information outside the active rolling window); requires modifying positional indexing during generation.

---

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)

- **Observations & Tricks Worth Stealing:**
  1. **Four Initial Tokens are All You Need for Softmax Stability:** The paper proves empirically across multiple model families (Llama-2, Falcon, MPT, Pythia) that retaining just four initial tokens is sufficient to anchor the attention denominator. Even replacing those tokens with meaningless line-break characters restores model stability, proving that their value is entirely structural rather than semantic. We should unconditionally pin the first 4–16 tokens of every session in GPU VRAM across all our benchmark policies.
  2. **Cache-Centric Positional Re-Indexing:** Discarding tokens from the middle creates artificial index gaps that cause positional extrapolation errors. StreamingLLM reassigns token positions consecutively based on their slot index within the physical cache buffer rather than their original sequence position. This trick allows models trained on short contexts (e.g., 4,000 tokens) to generate millions of tokens without out-of-distribution positional failures.
  3. **Pre-Training with a Dedicated Sink Token:** Pre-training models with an explicit placeholder token at the start of every sequence trains the network to dump all residual attention onto a single token, reducing the required sink budget from four tokens to just one.

- **The Catch / Blind Spots on Consumer Hardware:**
  - **Severe Middle-Context Amnesia:** StreamingLLM is strictly an infinite *recency streaming* technique, not an infinite *retrieval* engine. On benchmarks requiring recall of facts beyond the active window (such as long-span question answering or multi-session chat), accuracy immediately collapses to zero once the relevant fact rolls out of the cache.
  - **PagedAttention Block Table Misalignment:** Modern high-throughput engines like vLLM manage memory in discrete blocks (e.g., 16 tokens per block) and apply rotary position embeddings to keys at their absolute sequence positions during ingestion. Rolling individual tokens and dynamically re-indexing positions within the cache breaks standard contiguous block sharing and requires custom kernel adaptations.
  - **Evaluated on High-End Workstation Hardware (A6000):** The paper measured speedups on an NVIDIA RTX A6000 (48 GB VRAM). While the flat memory profile is ideal for consumer GPUs (RTX 4060 8 GB / RTX 4090 16 GB), consumer setups with limited VRAM must handle long-context recall, which StreamingLLM abandons entirely.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Our manual review notes (Row 19) accurately recognized the core structural mechanism: identifying the attention sink phenomenon, retaining initial sink tokens plus a sliding window, maintaining static memory over theoretically infinite decode length, and evaluating on Llama-2, MPT, Falcon, Pythia, PG19, and the deduplicated Pile.
  - *Confirmed Brainstorming Direction:* The manual review proposed anchoring $x$ sink tokens and $y$ sliding window tokens on the GPU while using priority-based offloading for the remaining tokens. This directly matches our project's 3-axis architecture: pinning the sink tokens and recent window in fast VRAM while offloading and quantizing the middle tokens rather than deleting them.
  - *Clarification from PDF:* The team notes stated that StreamingLLM makes the model "theoretically have infinite decode sequence length with static memory use." While true for local language fluency and perplexity, the full PDF clarifies that the model cannot recall past facts outside the window ($M$ tokens), scoring 0% on long-span question answering unless past tokens are explicitly retained or retrieved.

---

## 2. Metadata & Provenance

- **Authors & Affiliations:** Guangxuan Xiao (MIT), Yuandong Tian (Meta AI), Beidi Chen (CMU), Song Han (MIT & NVIDIA), Mike Lewis (Meta AI)
- **Publication / Venue & Year:** International Conference on Learning Representations (ICLR 2024), Oral Presentation. (arXiv:2309.17453v4, April 2024)
- **Primary Axis:** Structural (Attention sink token anchoring + sliding window rolling cache eviction)
- **Secondary Interacting Axis:** None (Pure structural token retention/eviction policy)
- **Official Codebase / Repository:** [https://github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)
- **Baseline Implementation Stack:** Built on PyTorch and HuggingFace Transformers (`transformers.models.llama`, `falcon`, `mpt`, `gpt_neox`). Evaluated across Llama-2 (7B, 13B, 70B), Falcon (7B, 40B), MPT (7B, 30B), and Pythia (2.8B, 6.9B, 12B), along with custom 160M pre-trained models on 8x NVIDIA RTX A6000 GPUs.

---

## 3. How the Algorithm Actually Works (Under the Hood)

- **Data Structures Added / Modified:**
  - **Dual-Region Rolling KV Cache Buffer:** The KV cache buffer for each attention head is partitioned into two contiguous segments with a total capacity of $C = x + y$:
    1. *Attention Sink Buffer ($\mathcal{C}_{\text{sink}}$):* A static buffer storing the Key and Value states of the first $x$ tokens (typically $x = 4$). These states are cached during prefill and never evicted or updated during decoding.
    2. *Rolling Window Buffer ($\mathcal{C}_{\text{recent}}$):* A circular FIFO queue holding the most recent $y$ tokens (e.g., $y = 1020$ or $2044$). When decoding token $t > C$, the oldest token in $\mathcal{C}_{\text{recent}}$ is evicted, and token $t$'s KV state is appended.
  - **Raw Key Storage Buffer:** For models using Rotary Position Embeddings (RoPE), Key vectors are stored *prior* to the application of the rotary embedding matrix ($R_{\theta, p}$) so that positions can be dynamically assigned relative to their cache position rather than their original text position.

- **Scoring, Compression & Decision Logic:**
  1. *Softmax Attention Sink Mechanism:*
     In standard multi-head self-attention, the attention distribution over cached tokens is computed as:
     $$\text{SoftMax}(s)_i = \frac{\exp(s_i)}{\sum_{j=1}^N \exp(s_j)} = \frac{\exp(s_i)}{\exp(s_1) + \sum_{j=2}^N \exp(s_j)}, \quad \text{where } s_i = \frac{q_t^\top k_i}{\sqrt{d_k}}$$
     *Plain English:* Every attention score is exponentiated and divided by the sum of all exponentiated scores so that they add up to 1. In deeper transformer layers, the model drives the score of the initial token $s_1 \gg s_j$, turning token 1 into a dominant term in the denominator. When standard window attention evicts token 1, the denominator loses this massive stabilizing component, causing the remaining attention weights to fluctuate wildly and wrecking generation.
  2. *Position Re-Assignment in Rolling RoPE:*
     Let the cached tokens at decoding step $t$ have original sequence indices $\{0, \dots, x-1\} \cup \{t-y, \dots, t-1\}$. Instead of using original sequence positions, StreamingLLM assigns cache-relative positional indices $p \in \{0, 1, \dots, x+y-1\}$:
     $$p_i = \begin{cases} i & \text{if } i < x \text{ (sink tokens)} \\ x + (i - x) & \text{if } i \ge x \text{ (recent window tokens)} \end{cases}$$
     For a query token at decode step $t$, its effective relative distance to token $i$ in the cache is calculated as:
     $$\text{AttnScore}(q_t, k_i) = \frac{\left(R_{\theta, \, x+y} \, q_t\right)^\top \left(R_{\theta, \, p_i} \, k_i^{\text{raw}}\right)}{\sqrt{d_k}}$$
     *Plain English:* Rather than letting the positional index grow to 10,000 or 1,000,000 (which exceeds the model's pre-trained position range), the tokens inside the cache are relabeled sequentially from 0 up to the cache size ($x+y$). The query is given position $x+y$, ensuring that all relative distances remain strictly within the model's pre-trained attention window.
  3. *Pre-Training with Learnable Sink or Zero Sink:*
     To eliminate reliance on multiple ad-hoc initial tokens, the paper explores SoftMax-off-by-one ($\text{SoftMax}_1$):
     $$\text{SoftMax}_1(s)_i = \frac{\exp(s_i)}{1 + \sum_{j=1}^N \exp(s_j)}$$
     *Plain English:* Adding a constant 1 to the denominator allows the attention weights to sum to less than 1. This is mathematically equivalent to prepending a virtual token whose Key and Value vectors are all zeros ("Zero Sink"). Alternatively, prepending a dedicated, trainable token placeholder (`[SINK]`) at position 0 during pre-training allows the model to concentrate all unneeded attention onto exactly one dedicated token ($x=1$).
  - **Execution Cadence:** The 4 sink tokens are written once during initial prefill. Rolling eviction and cache-relative rotary position assignment execute at every single decode token step ($O(1)$ memory, $O(L)$ decode compute per token).

- **Memory Hierarchy & Placement:**
  - **Keys ($K$) & Values ($V$):** Stored in native BF16 or FP16 precision in GPU VRAM. No quantization or offloading is used.
  - **Placement:** 100% of the active cache ($\mathcal{C}_{\text{sink}} \cup \mathcal{C}_{\text{recent}}$) resides strictly in GPU VRAM. Evicted tokens are discarded entirely.
  - **Granularity:** Token-level circular FIFO replacement within the rolling buffer.

- **Decode-Phase Overheads:**
  - **PCIe / CPU Transfer Overhead:** Exactly $0\text{ s}$. The cache size is fixed and resident in GPU memory; no host-device transfers occur during generation.
  - **Computational Overhead:** Requires applying the rotary position embedding to raw Key vectors at attention time, or shifting positional biases (in ALiBi). Decoding complexity per token is strictly $O(C)$ where $C = x+y$, completely avoiding the $O(T^2)$ re-computation cost of sliding window generation.

---

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)

- **Memory Saved (Efficiency):**
  - **Static Footprint:** GPU memory usage remains strictly flat across indefinite sequence lengths, scaling only with the fixed cache size $C = x + y$:
    - Llama-2-7B (Cache 2048): VRAM usage remains flat at $\approx 13.5\text{ GB}$ throughout 4 million tokens, compared to Full Attention which would require over $1\text{ TB}$ of KV cache memory.
    - Llama-2-13B (Cache 2048): VRAM usage remains flat at $\approx 26.2\text{ GB}$.
  - Compared to the Sliding Window with Re-computation baseline (which also maintains a bounded window), StreamingLLM uses virtually identical memory during steady-state decoding ($\approx 13\text{--}14\text{ GB}$ for 7B at cache 256–2048; $\approx 16\text{ GB}$ at cache 4096 vs. $19\text{ GB}$ for re-computation due to intermediate prefill activation peaks).

- **Speed & Latency (TTFT, TPOT, Tail Latency):**
  - **Per-Token Decoding Latency vs. Sliding Window Re-Computation (NVIDIA RTX A6000, FP16):**
    - *Llama-2-7B:*
      - Cache 256: Re-computation $63\text{ ms}$ vs. StreamingLLM $31\text{ ms}$ ($2.0\times$ speedup).
      - Cache 1024: Re-computation $223\text{ ms}$ vs. StreamingLLM $35\text{ ms}$ ($6.4\times$ speedup).
      - Cache 2048: Re-computation $523\text{ ms}$ vs. StreamingLLM $45\text{ ms}$ ($11.6\times$ speedup).
      - Cache 4096: Re-computation $1411\text{ ms}$ vs. StreamingLLM $65\text{ ms}$ ($21.7\times$ speedup).
    - *Llama-2-13B:*
      - Cache 256: Re-computation $99\text{ ms}$ vs. StreamingLLM $48\text{ ms}$ ($2.1\times$ speedup).
      - Cache 1024: Re-computation $361\text{ ms}$ vs. StreamingLLM $60\text{ ms}$ ($6.0\times$ speedup).
      - Cache 2048: Re-computation $860\text{ ms}$ vs. StreamingLLM $75\text{ ms}$ ($11.5\times$ speedup).
      - Cache 4096: Re-computation $2355\text{ ms}$ vs. StreamingLLM $106\text{ ms}$ (**$22.2\times$ speedup**).
  - **Tail Latency (P95/P99):** Not reported in paper (mean per-token decoding latency only).

- **Output Quality & Trustworthiness:**
  - **Language Modeling Perplexity on PG-19 (First Book, 65K tokens, Llama-2-13B):**
    - Full Dense Attention: $\text{PPL} = 5641.0$ (collapses past training length 4K).
    - Window Attention without Sinks ($0 + 1024$): $\text{PPL} = 5158.07$ (catastrophic collapse).
    - Sliding Window with Re-Computation (Oracle): $\text{PPL} = 5.43$.
    - StreamingLLM ($4 + 1020$): $\text{PPL} = 5.40$ (fully matches oracle performance).
    - Replacing Initial 4 Tokens with `\n` ($4\text{"\textbackslash n"} + 1020$): $\text{PPL} = 5.60$ (proves sink phenomenon is positional/structural, not semantic).
  - **Ablation on Sink Token Count ($x + y$ on PG-19 Concatenated 400K tokens):**
    - Llama-2-7B ($C=4096$): $0+4096 \to 3359.95$; $1+4095 \to 11.88$; $2+4094 \to 10.51$; $4+4092 \to 9.59$; $8+4088 \to 9.54$.
    - Falcon-7B ($C=2048$): $0+2048 \to 17.90$; $1+2047 \to 12.12$; $4+2044 \to 12.12$.
    - MPT-7B ($C=2048$, ALiBi): $0+2048 \to 460.29$; $1+2047 \to 14.99$; $4+2044 \to 14.99$.
    - Pythia-12B ($C=2048$): $0+2048 \to 21.62$; $1+2047 \to 11.95$; $4+2044 \to 12.09$.
  - **Extreme Sequence Scaling:**
    - Evaluated across Llama-2 (7B, 13B, 70B), Falcon (7B, 40B), MPT (7B, 30B), and Pythia (2.8B, 6.9B, 12B) on concatenated PG-19 text up to **4,000,000 tokens**; perplexity remains completely stable without divergence.
  - **Multi-Round Question Answering (ARC-Easy / ARC-Challenge Streaming):**
    - Llama-2-7B-Chat: One-shot sample-by-sample baseline: $71.25\% / 53.16\%$. Window attention ($0+1024$): $3.58\% / 1.39\%$. Dense attention: OOM. StreamingLLM ($4+1020$): $71.34\% / 55.03\%$ (preserves full one-shot conversational accuracy).
    - Llama-2-70B-Chat: One-shot baseline: $91.29\% / 78.50\%$. Window attention: $0.12\% / 0.32\%$. StreamingLLM: $91.37\% / 80.20\%$.
  - **StreamEval Retrieval Limits (Llama-2-7B-32K-Instruct, Table 7):**
    - At line distance 20 ($460$ tokens): accuracy is $85.80\%$ ($4+2044$) and $84.60\%$ ($4+4092$).
    - At line distance 100 ($2300$ tokens): accuracy drops to $0.00\%$ on $4+2044$ cache (because the fact was evicted from the rolling window), while $4+4092$ retains $61.60\%$.
    - Once query-answer distance exceeds the cache size, accuracy drops to **$0.00\%$**, confirming that StreamingLLM does not extend long-term retrieval memory.
  - **LongBench Evaluation (Llama-2-7B-Chat, Table 8):**
    - Standard Middle-Truncation baseline ($1750+1750$ tokens): HotpotQA $25.4\%$, NarrativeQA $18.7\%$, GovReport $27.3\%$.
    - StreamingLLM ($4+3496$): HotpotQA drops to $21.6\%$, NarrativeQA drops to $11.6\%$, GovReport drops to $23.9\%$ due to loss of early document context.
    - StreamingLLM ($1750+1750$ sink/recent split): HotpotQA $24.9\%$, NarrativeQA $18.2\%$, GovReport $26.3\%$ (matches truncation baseline).

---

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)

- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16 Weights Rule:** Fully compliant. StreamingLLM requires zero fine-tuning and zero modification of model weights.
  - **Dense GQA Compatibility:** Fully compatible with our pinned Grouped-Query Attention models (**Qwen 3 4B**, **Phi-4 mini**, **Ministral 3 3B**, **SmolLM3 3B**, **Qwen 3 1.7B**). Because GQA groups multiple query heads to share a single key-value head, retaining the first 4 tokens preserves the attention sink across all query heads simultaneously.
  - **Hybrid & SWA Models:** Provides direct theoretical grounding for models that natively employ Sliding Window Attention (such as **Gemma 4 E4B Instruct**), explaining why architecture-level sliding windows must always preserve initial tokens or global sink tokens.

- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - **Local Desktop (8 GB RTX 4060):**
    - Running **Qwen 3 1.7B** (BF16 weights: $3.45\text{ GB}$): A cache budget of $C = 2048$ ($112\text{ KiB/token}$) consumes only $0.22\text{ GB}$ of VRAM. Total memory remains at $\approx 3.7\text{ GB}$, leaving over $4\text{ GB}$ of VRAM completely free and guaranteeing zero OOM across infinite streaming tokens.
  - **Slurm HPC Cluster (16 GB RTX 4090 Cap):**
    - Running **Qwen 3 4B Instruct** (BF16 weights: $8.06\text{ GB}$): A cache budget of $C = 4096$ ($144\text{ KiB/token}$) consumes $0.56\text{ GB}$ per session. On our $16\text{ GB}$ cap, this allows serving **12 to 14 concurrent streaming sessions** indefinitely without memory exhaustion, fully satisfying the SME multi-tenancy requirements outlined in `manual_reviews.md`.

- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** Medium. Pure Python / PyTorch implementation for baseline validation is straightforward. However, integrating into production engines (`vLLM` / `LMCache`) involves structural constraints.
  - **WP-3 Micro-Benchmarking:**
    - Can be immediately incorporated into `kv_bench` as a structural baseline to benchmark the trade-off between KV cache footprint, decoding throughput, and memory retention on `StreamEval` / `NIAH`.
  - **WP-6 Engine Integration (`vLLM` PagedAttention & `LMCache`):**
    - *The PagedAttention RoPE Conflict:* In standard vLLM execution, Key vectors are rotated using absolute sequence positions ($t$) before being written to physical block tables. Rolling individual tokens and shifting their positional embedding relative to the cache index conflicts with vLLM's immutable block caching.
    - *Adaptation for Our 3-Axis Engine:* Rather than throwing evicted middle tokens into the void (which causes the $0\%$ retrieval failure on long documents), our project architecture adapts StreamingLLM as follows:
      1. **Sink Pinning:** Permanently pin the first 16 tokens (the first PagedAttention block) in GPU VRAM in native BF16.
      2. **Rolling Local Window:** Maintain a high-priority local attention window (e.g., 2048 tokens) in GPU VRAM.
      3. **Tiered Offload & Quantization:** Instead of evicting middle tokens, route evicted blocks through **LMCache** to host CPU RAM with 4-bit quantization (e.g., via `LMCacheMPConnector`), preserving long-term conversational recall while maintaining the throughput advantages of attention sink stabilization.

- **Classification for Project:** **Core Foundational Reference & Structural Component Inspiration** (Establishes the indispensable requirement of pinning initial sink tokens in all eviction, quantization, and offloading policies across WP-3 and WP-4).
