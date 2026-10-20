# The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference

## 0. Plain-English TL;DR (What It Is & Why It Matters)

- **The Big Picture:**
  In modern artificial intelligence serving, engineers have always assumed that using a Key-Value cache produces the exact same answers as recalculating past words from scratch, only much faster. This landmark paper proves that assumption is completely false when models run in standard sixteen-bit floating-point precision. Because computers round numbers slightly differently depending on the order in which additions are performed, a model using a cache and a model recomputing everything from scratch will eventually diverge and produce completely different text sequences one hundred percent of the time, even when choosing words deterministically with zero randomness.

- **Core Concepts Explained Simply:**
  - **The Rounding Order Problem:** When a computer adds numbers together in sixteen-bit precision, adding the first two numbers and then adding a third produces a slightly different result than adding the last two first. In technical terms, sixteen-bit computer math is non-associative.
  - **Cache Versus Full Recomputation:** When the cache is turned on, the graphics processor calculates the attention for a single new word and appends it to stored memory. When the cache is turned off, the processor recalculates the entire conversation history as one giant block. These two paths execute math operations in a different order, introducing microscopic rounding discrepancies.
  - **Compounding Memory Drift:** These tiny rounding differences are written directly into the stored Key-Value memory at every single step. Instead of washing out, they accumulate across neural network layers and across generated words, permanently warping future attention calculations.
  - **Decision Boundary Flips:** Small numerical drifts cause words to change whenever the model is unsure between two equally good word choices. At these uncertain decision points, even the smallest rounding difference causes the model to pick a different word, leading to a completely different generated response.

---

## 1. Key Takeaways for Our Project

- **What This Means for Our 8–16 GB Consumer GPUs:**
  - **Baseline Definition for Evaluation:** In our benchmarking suite, we cannot evaluate the quality of our Key-Value cache compression policies by comparing them against a cache-free recomputed baseline. If we did, normal sixteen-bit math rounding differences would create massive divergence that could easily be mistaken for compression artifacts. We must strictly measure all fidelity metrics against the uncompressed, cached sixteen-bit baseline.
  - **Compounded Vulnerability in BF16:** Our pinned models run in native Brain Floating Point sixteen format. Brain floating point has only seven precision bits compared to standard sixteen-bit floating point which has ten precision bits. This means computer rounding errors are eight times coarser in our native precision, making our consumer models even more prone to execution-path divergence.
  - **Error Broadcasting in Grouped-Query Attention:** Every single model pinned in our project uses Grouped-Query Attention, where several attention heads share a single Key-Value head. As this paper shows, when multiple heads share the same cached memory, a single rounding error is copied to all attending heads simultaneously, magnifying the drift across network layers.

- **Formulas & Rules of Thumb to Keep:**
  - **Rule of Thumb on Baseline Ground Truth:** Always compare compressed Key-Value cache runs against an uncompressed cached execution path rather than a recomputed execution path, because cache-free and cached generation naturally diverge one hundred percent of the time under sixteen-bit precision.
  - **Rule of Thumb on High-Entropy Tokens:** Expect token selection flips to happen primarily on ambiguous, high-uncertainty words near decision boundaries. A drop in word agreement does not automatically mean reasoning failure; it often indicates boundary shifts at ambiguous transition words.
  - **Rule of Thumb on Grouped-Query Amplification:** The higher the ratio of query heads to Key-Value heads, the faster rounding errors propagate through the model, because a single stored error affects multiple attention computations at once.

- **Verification of Team Notes (`manual_reviews.md`):**
  - **Confirmed:** Row 30 in our team notes correctly identified the paper's core thesis: Key-Value caching is not numerically equivalent to cache-free recomputation under sixteen-bit precision due to non-associativity, leading to a one hundred percent token divergence rate even under deterministic greedy decoding.
  - **Confirmed Relevance:** The team notes correctly identified that this finding is essential for Work Package Six and our project proposal Section 3.3.2, justifying why we must track logit divergence and top-one token agreement while establishing the proper uncompressed baseline.
  - **Refinement from Document:** The team notes did not capture three critical architectural findings revealed in the paper:
    1. Switching to thirty-two-bit floating point completely eliminates the divergence (dropping flips to exactly zero percent and reducing distribution drift by eight orders of magnitude), proving arithmetic order is the sole causal driver.
    2. Grouped-Query Attention causes an immediate, massive error surge in the very first layer (multiplying drift by thirty-five to one hundred times), whereas models with larger head dimensions and sliding window attention accumulate errors gradually and evenly across all layers.
    3. Attempting to fix the divergence by intervening on the neural network hidden activations completely fails, proving that the error is trapped inside the stateful Key-Value memory itself.

---

## 2. System & Architectural Analysis

- **Core Contribution:**
  - Establishes the first systematic empirical and mechanistic proof that Key-Value (KV) caching is **not** numerically equivalent to cache-free recomputation under standard half-precision floating-point arithmetic ($\text{FP16}$).
  - Evaluates 31,500 run pairs across three open-weight architectures ($\text{LLaMA-2-7B}$, $\text{Mistral-7B-v0.3}$, and $\text{Gemma-2-2B}$) on the $\text{GSM8K}$ multi-step mathematical reasoning benchmark across three decoding regimes (deterministic greedy decoding, top-$k$ sampling, and nucleus top-$p$ sampling).
  - Proves that the divergence is deterministic, reproducible, 100% universal across all model-strategy conditions, and causally driven by floating-point non-associativity in accumulation order.
  - Disproves the assumption that divergence is carried by the transformer residual stream by showing that residual stream activation patching fails to recover the cache-free trajectory, localizing the causal variable strictly to the stateful KV cache tensors.

- **Architectural Bottlenecks / Numerical Failure Modes Addressed:**

  1. **Floating-Point Non-Associativity and Accumulation Order:**
     IEEE 754 floating-point addition and multiplication are non-associative operations due to finite mantissa truncation and rounding:
     $$\begin{aligned}
     (a + b) + c &\neq a + (b + c) \\
     (a \times b) \times c &\neq a \times (b \times c)
     \end{aligned}$$
     *Plain English translation:* In standard floating-point hardware arithmetic, adding or multiplying numbers in a different sequence produces different rounded bit patterns. In transformer inference, the primary driver is addition non-associativity during large inner dot-product reductions.
     
     In cache-free recomputation ($\text{cache-OFF}$), the self-attention operation at decode step $t$ computes the full matrix multiplication over the entire prefix sequence $[1, \dots, t]$ simultaneously within a single large General Matrix Multiply ($\text{GEMM}$) kernel:
     $$A_{\text{OFF}} = \text{Softmax}\left(\frac{Q_{1:t} K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$
     *Plain English translation:* The cache-free path reprojects and multiplies all tokens from the prompt up to step $t$ together in one large parallel matrix block.
     
     In cached inference ($\text{cache-ON}$), the attention operation is split into historical cached states and an incremental vector multiplication ($\text{GEMV}$) for the single arriving token:
     $$A_{\text{ON}} = \text{Softmax}\left(\frac{q_t [K_{1:t-1}; k_t]^T}{\sqrt{d_k}}\right) [V_{1:t-1}; v_t]$$
     *Plain English translation:* The cached path stores historical Keys and Values in memory buffers and only computes new vectors for token $t$, concatenating them before computing attention.
     
     Because parallel GPU reduction trees accumulate the dot products across different tile boundaries and memory layouts, the summation order differs between the two paths. The authors show that the relative accumulation error per inner dot product is constant:
     $$\epsilon_{\text{accum}} \approx 0.036\% \quad (\sigma = 8.5 \times 10^{-7})$$
     *Plain English translation:* Every single dot product calculation introduces an average relative error of roughly 0.036 percent, which remains flat regardless of input prompt length or dataset.

  2. **Grouped-Query Attention (GQA) Error Broadcasting:**
     Modern LLMs employ Grouped-Query Attention ($\text{GQA}$) with a sharing ratio $R = H_Q / H_{KV}$, where $H_Q$ query heads share $H_{KV}$ Key-Value heads ($R = 1$ in standard Multi-Head Attention, $R = 4$ in Mistral-7B, $R = 2$ in Gemma-2, and $R = H_Q$ in Multi-Query Attention).
     
     Under $\text{cache-ON}$, any rounding error $\delta K$ or $\delta V$ written into a shared KV tensor is broadcast to all $R$ query heads attending to that group:
     $$\text{Attn}^{(h)} = \text{Softmax}\left(\frac{Q^{(h)} (K + \delta K)^T}{\sqrt{d_k}}\right) (V + \delta V), \quad \forall h \in \{1, \dots, R\}$$
     *Plain English translation:* In Grouped-Query Attention, when a Key or Value vector has a rounding error, that exact same error is injected into all $R$ query heads at the same time, turning what would have been independent head errors into correlated, amplified errors.
     
     This produces distinct, architecturally predictable layer drift profiles:
     - **$\text{LLaMA-2-7B}$ ($\text{MHA}, R = 1$, head dim $d_k = 128$):** Hidden state drift ($\|h_{\text{ON}} - h_{\text{OFF}}\|_2$ at decode step 1) surges from $3.45$ at layer 0 to $361.96$ at layer 1 ($105\times$ amplification). Cosine similarity drops to $0.49$.
     - **$\text{Mistral-7B-v0.3}$ ($\text{GQA}, R = 4$, head dim $d_k = 128$):** Hidden state drift explodes from $25.76$ at layer 0 to $902.36$ at layer 1 ($35\times$ amplification to the highest absolute drift). Cosine similarity collapses to $0.015$ (near-complete decorrelation at layer 0).
     - **$\text{Gemma-2-2B}$ ($\text{GQA}, R = 2$, head dim $d_k = 256$, sliding window attention):** Exhibits a flat drift profile ($1.0 \text{ to } 1.4$ across all 26 layers) with no single-layer spike. However, due to its larger head dimension and sliding window localization, errors accumulate uniformly across all layers, resulting in the highest total behavioral divergence.

  3. **Controlled $\text{FP32}$ Falsification:**
     To confirm that floating-point non-associativity is the sole root cause, the authors evaluated an $\text{FP32}$ inference pipeline across $n = 600$ examples per model:
     
     | Model | $\text{FP16}$ Mean $\text{KL}$ | $\text{FP32}$ Mean $\text{KL}$ | $\text{FP16}$ Flip Rate | $\text{FP32}$ Flip Rate |
     | :--- | :---: | :---: | :---: | :---: |
     | **$\text{LLaMA-2-7B}$** | $0.113$ | $2.80 \times 10^{-8}$ | $1.15\%$ | **$0.0\%$** |
     | **$\text{Mistral-7B-v0.3}$** | $0.122$ | $2.50 \times 10^{-8}$ | $1.14\%$ | **$0.0\%$** |
     | **$\text{Gemma-2-2B}$** | $0.112$ | $2.78 \times 10^{-8}$ | $1.03\%$ | **$0.0\%$** |
     
     *Key Finding:* Under $\text{FP32}$, output distribution divergence collapses to a hardware noise floor of $\approx 2.5 \times 10^{-8}$ (a reduction exceeding eight orders of magnitude, $> 4 \times 10^8$), and the token flip rate drops to exactly $0.0\%$. This falsifies alternative hypotheses (e.g., algorithmic bugs, kernel nondeterminism, attention sink disparities) and isolates $\text{FP16}$ accumulation non-associativity as the sole causal driver.

  4. **Decision Boundary Proximity & Model Uncertainty:**
     The paper investigates why examples that experience early token flips (flip index $t \le 1$) also exhibit significantly higher vocabulary Kullback-Leibler ($\text{KL}$) divergence:
     $$\text{KL}(p_t \parallel q_t) = \sum_{v \in \mathcal{V}} p_t(v) \log \left(\frac{p_t(v)}{q_t(v)}\right)$$
     *Plain English translation:* Kullback-Leibler divergence sums the probability differences between the cached distribution $p_t$ and the cache-free distribution $q_t$ across every word $v$ in the entire model vocabulary $\mathcal{V}$.
     
     $$\text{JS}(p_t \parallel q_t) = \frac{1}{2} \text{KL}(p_t \parallel m_t) + \frac{1}{2} \text{KL}(q_t \parallel m_t), \quad m_t = \frac{1}{2}(p_t + q_t)$$
     *Plain English translation:* Jensen-Shannon divergence is a balanced, symmetric version of Kullback-Leibler divergence bounded between zero and one, used to confirm that divergence trends are not mathematical artifacts of unbounded ranges.
     
     The authors discover a statistically significant negative Pearson correlation between flip index and mean $\text{KL}$ divergence across all models:
     - $\text{LLaMA-2}$: $r = -0.202$ ($p < 10^{-96}$)
     - $\text{Mistral}$: $r = -0.152$ ($p < 10^{-55}$)
     - $\text{Gemma}$: $r = -0.120$ ($p < 10^{-34}$)
     
     *Resolution:* Early flips and high $\text{KL}$ are joint consequences of model uncertainty rather than error accumulation speed. When a model operates near a decision boundary (competing tokens have near-identical logit margins), two things occur simultaneously:
     - The output distribution spreads entropy across multiple candidate tokens (inflating full-vocabulary $\text{KL}$).
     - Microscopic numerical perturbations easily tip the argmax selection, triggering an early token flip (median flip index across all models is $0$, meaning flips occur on the very first decoded token).
     - When early boundary flips ($\text{flip} \le 3$) are excluded, the negative correlation strengthens substantially for $\text{GQA}$ models ($\text{Mistral}$: $-0.152 \to -0.333$; $\text{Gemma}$: $-0.120 \to -0.213$).

  5. **Causal Localization via Activation Patching:**
     To verify whether the numerical drift resides in the transformer residual stream or in the stateful KV cache, the authors perform activation patching on the $n = 600$ highest-$\text{KL}$ examples. Cache-ON hidden states $h_{\text{ON}}$ are injected into the cache-OFF forward pass at decode step 0:
     $$\text{Recovery \%} = \frac{\text{KL}_{\text{base}} - \text{KL}_{\text{patched}}}{\text{KL}_{\text{base}}} \times 100$$
     *Plain English translation:* Measures what percentage of distribution difference is erased when hidden states from the cached execution path are forced into the cache-free model.
     
     *Empirical Patching Results:*
     - Single-layer residual patching yielded a net median recovery of $-0.019\%$ (mean $-1.21\%$).
     - Simultaneous cumulative patching of all layers across the entire residual stream produced negative recovery in all models ($\text{LLaMA-2}$: $-5.34\%$, $\text{Mistral}$: $-2.56\%$, $\text{Gemma}$: $-2.88\%$).
     - Patching the residual stream increases divergence rather than repairing it. This establishes a definitive null result: the causal variable does not reside in attention projection weights or feed-forward activations, but is locked inside the historical KV tensors that continuously feed past rounding errors into every future attention step.

- **Relevance to `kv_bench` Baseline Work Packages (`WP-1`, `WP-2`, `WP-6`):**

  1. **Strict Definition of Ground-Truth Baseline (`WP-1` & `WP-6`):**
     - Prior KV cache research occasionally compares compressed or evicted caches against a cache-free recomputed forward pass, or assumes that uncompressed cached inference is numerically identical to recomputation.
     - Chodavarapu et al. prove that comparing any cached policy against a cache-free recomputed baseline guarantees a $100\%$ token divergence rate purely from $\text{FP16}$ non-associativity.
     - **Mandate for `kv_bench`:** In all evaluation work packages, the uncompressed baseline **must** be the uncompressed $\text{cache-ON}$ execution path under identical runtime and precision conditions, **never** a cache-free recomputation path.

  2. **Implications for Pinned Models in Native $\text{BF16}$ ([`docs/MODELS.md`](../../MODELS.md)):**
     - Our pinned model suite comprises:
       - `Qwen 3 4B` ($8:1 \text{ GQA}$, native $\text{BF16}$)
       - `Phi-4 mini` ($3.8\text{B}$, $8:1 \text{ GQA}$, native $\text{BF16}$)
       - `Ministral 3 3B` ($4:1 \text{ GQA}$, native $\text{BF16}$)
       - `SmolLM3 3B` ($4:1 \text{ GQA}$, native $\text{BF16}$)
       - `Qwen 3 1.7B` ($4:1 \text{ GQA}$, native $\text{BF16}$)
     - **Floating-Point Mantissa Truncation:** While Chodavarapu et al. studied $\text{FP16}$ (which has 1 sign bit, 5 exponent bits, and 10 mantissa bits; machine epsilon $\epsilon_{\text{FP16}} = 2^{-10} \approx 9.77 \times 10^{-4}$), our pinned models run in native $\text{BF16}$ (1 sign bit, 8 exponent bits, and only 7 mantissa bits; machine epsilon $\epsilon_{\text{BF16}} = 2^{-7} \approx 7.81 \times 10^{-3}$).
     - Because $\text{BF16}$ has an $8\times$ coarser machine epsilon than $\text{FP16}$, non-associative accumulation errors during attention dot products are significantly amplified on consumer hardware running native $\text{BF16}$.
     - Furthermore, all our pinned models utilize aggressive Grouped-Query Attention ($4:1$ or $8:1 \text{ GQA}$), which systematically broadcasts initial KV rounding errors across 4 to 8 query heads simultaneously, replicating the sharp layer-1 decorrelation observed in Mistral.

  3. **Diagnostic Metric Suite Alignment ([`docs/proposal.md`](../../proposal.md) Section 3.3.2):**
     - Section 3.3.2 requires measuring three fidelity dimensions:
       1. **Per-Token Final-Logit $\text{KL}$ Divergence:** Validated by Chodavarapu et al. as the superior continuous metric for tracking probability mass redistribution prior to argmax sampling.
       2. **Top-1 Token Agreement Rate:** Captures hard behavioral flips. Chodavarapu et al. demonstrate that Top-1 flips occur primarily at high-entropy decision boundaries (median flip index $= 0$). A drop in Top-1 agreement must be cross-analyzed with model output entropy before attributing it to policy failure.
       3. **Perplexity Delta ($\Delta\text{PPL}$):** Must be computed over identical uncompressed cached forward passes.
     - **Hardware Feasibility (8 GB RTX 4060 & 16 GB RTX 4090):**
       Running dual forward passes (e.g., cached vs cache-free) during profiling requires double the compute and memory. By proving that the uncompressed $\text{cache-ON}$ execution path is the only valid reference, `kv_bench` avoids the severe latency and VRAM penalty of recomputation, preserving full consumer GPU memory for testing 3-axis eviction, selective quantization, and PCIe offloading.
