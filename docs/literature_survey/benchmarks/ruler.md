# RULER: What’s the Real Context Size of Your Long-Context Language Models?

## 0. Plain-English TL;DR (What It Tests & Why We Use It)

- **What This Benchmark Is:**
  RULER is a synthetic diagnostic benchmark developed by NVIDIA researchers to test the true usable context window of long-context language models. Instead of taking advertised context limits (such as 32,000 to 1 million tokens) at face value, RULER probes models with controllable sequence lengths and varying task difficulties to find the exact point where comprehension breaks down.

- **Why Standard Tests Aren't Enough:**
  - Simple **Needle In A Haystack (NIAH)** and passkey tests create a false sense of security: almost every modern long-context language model scores near 100% on them.
  - Retrieving a single hidden number from a static essay is too easy. Vanilla tests do not evaluate tracking entity connections across multiple steps, filtering out confusing distractor needles, aggregating facts spread across thousands of words, or answering complex questions with subtle clues.
  - Simple **Perplexity ($\Delta\text{PPL}$)** checks only measure overall token predictability, completely missing whether an eviction policy silently dropped a specific variable assignment or fact.

- **How It Works Step-by-Step:**
  1. *Controllable Synthetic Generation:* Rather than relying on static web documents that models may have memorized during pre-training, RULER automatically generates synthetic test inputs at custom lengths ($4\text{K}, 8\text{K}, 16\text{K}, 32\text{K}, 64\text{K}, 128\text{K}$ tokens) with mathematically controlled complexity.
  2. *Four Diagnostic Task Domains:*
     - **Retrieval:** Single needle, multi-keys (distractor needles), multi-values (multiple facts per key), and multi-queries (retrieving several distinct facts at once).
     - **Multi-hop Tracing:** Variable tracking where variables are assigned to each other in a chain, testing whether the model can trace connected entities through long text.
     - **Aggregation:** Extracting frequent or common words from long word sequences, testing long-range information synthesis.
     - **Question Answering:** Single-hop (SQuAD) and multi-hop (HotpotQA) questions hidden within distracting text, evaluating realistic text matching.
  3. *Effective Context Sizing:* The benchmark establishes a quality baseline (a standard 7B model at 4,000 tokens). A model's "effective context size" is defined as the longest sequence length at which it still maintains performance above this threshold.

---

## 1. Key Takeaways & How We Use It in `kv_bench`

- **Specific KV-Cache Failures Exposed:**
  - **Distractor Vulnerability Under Key Quantization:** When key vectors are quantized or compressed too aggressively, attention routing blurs. In multi-key tests with distractors, models frequently retrieve values from distractor keys or nearby text rather than the target needle.
  - **Chain Severing in Multi-Hop Tracing:** Eviction policies that discard tokens based solely on immediate attention scores discard intermediate variable assignments (e.g., $X_2 = X_1$) before the final query even arrives, breaking the reasoning chain completely.
  - **Multi-Value Recall Blind Spots:** When a single key has multiple associated values, compressed caches frequently output duplicate answers or miss secondary values, showing that memory compression disproportionately hurts multi-fact recall.
  - **Degradation into Verbatim Copying:** Under high context load, models frequently fail by simply copying the prompt instructions or prompt prefixes verbatim, or falling back on pre-trained parametric knowledge instead of reading their cache.

- **Key Findings from the Paper Worth Knowing:**
  1. **Advertised Context Limits Are Greatly Exaggerated:** Among 17 evaluated models claiming context windows of $32\text{K}$ to $1\text{M}$ tokens, only half maintained acceptable performance at $32\text{K}$ tokens. Almost all open-source models experienced severe accuracy cliffs long before reaching their advertised limits.
  2. **Vanilla NIAH Masks Failure:** Models that achieved flawless $100\%$ accuracy on vanilla passkey and single-needle retrieval dropped by $20\%\text{--}50\%$ on multi-hop tracing and aggregation as context length scaled.
  3. **Non-Transformer Architectures Lagged Significantly:** Recurrent and state-space architectures (Mamba-2.8B, RWKV-v5-7B) degraded sharply beyond $4\text{K}$ tokens, trailing standard Transformer baselines by wide margins on complex long-context tasks.
  4. **Model Size Strongly Aids Long-Context Robustness:** Within the same model family and training context length, larger parameter counts (e.g., Yi-34B vs. Yi-6B) demonstrated markedly better resistance to distractors and entity-tracing failures.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 22 of `manual_reviews.md` accurately identified RULER's expansion beyond vanilla NIAH into four task categories (single/multi-needle retrieval, multi-query retrieval, word aggregation, and variable assignment tracking chains) and noted that models experience severe accuracy cliffs well before claimed limits.
  - *Refinements from PDF Ground Truth:* The team skimming note listed SQuAD, HotpotQA, and Paul Graham essays as separate datasets. The PDF clarifies their architectural roles: Paul Graham essays and repeated sentences serve as synthetic *haystack backgrounds* for retrieval tasks, SQuAD and HotpotQA serve as the *QA domain*, while word distributions and variable assignment chains form the *Aggregation* and *Multi-hop Tracing* domains. Furthermore, in our project proposal (Section 3.3.4), RULER is specifically adopted as a targeted **Diagnostic Metric** (with synthetic word-counting subtasks omitted to focus exclusively on dialogue-relevant retrieval and tracking).

---

## 2. Workload & Diagnostic Profile

- **Publication & Authors:** Cheng-Ping Hsieh*, Simeng Sun*, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg (NVIDIA). Published at COLM 2024 (Conference on Language Modeling). (arXiv:2404.06654v3)
- **Target Workload & Context Lengths:**
  - Evaluated sequence lengths: $4\text{K}, 8\text{K}, 16\text{K}, 32\text{K}, 64\text{K}, 128\text{K}$ tokens (extended up to $256\text{K}$ on Yi-34B).
  - 500 automatically generated synthetic examples per task per length configuration.
  - Evaluation protocol: Greedy decoding in BF16 on 8x NVIDIA A100 GPUs using vLLM. Answer prefixes are appended to model responses to prevent refusal and ensure deterministic recall-based grading.

- **The Four Diagnostic Categories & 13 Evaluated Tasks:**
  1. *Retrieval (8 subtasks):*
     - **Single NIAH (S-NIAH):**
       - *Subtask 1 (Passkey Retrieval):* Word key, number value (7 digits), embedded in repeated noise sentences ("The grass is green. The sky is blue...").
       - *Subtask 2 (Vanilla NIAH):* Word key, number value, embedded in Paul Graham essays.
       - *Subtask 3 (UUID NIAH):* Word key, 32-digit UUID value, embedded in Paul Graham essays.
     - **Multi-Keys NIAH (MK-NIAH):**
       - *Subtask 1:* 4 key-value pairs inserted into essays; query asks for 1 key (tests ignoring 3 hard distractors).
       - *Subtask 2 (Line Retrieval):* Haystack filled entirely with distractor needles; query asks for 1 target number.
       - *Subtask 3 (KV Retrieval):* Haystack filled entirely with distractor needles using 32-digit UUID keys and values.
     - **Multi-Values NIAH (MV-NIAH):** 1 key associated with 4 distinct numbers scattered across essays; query requires returning all 4 values.
     - **Multi-Queries NIAH (MQ-NIAH):** 4 distinct key-value pairs scattered across essays; query requires retrieving the values for all 4 distinct keys simultaneously.
  2. *Multi-hop Tracing (1 task):*
     - **Variable Tracking (VT):** 1 chain with 4 name-binding hops (e.g., $\text{VAR } X_1 = V, \text{VAR } X_2 = X_1, \text{VAR } X_3 = X_2, \text{VAR } X_4 = X_3, \text{VAR } X_5 = X_4$), requiring the model to identify all 5 variable names bound to value $V$.
  3. *Aggregation (2 tasks):*
     - **Common Words Extraction (CWE):** 10 common words (frequency $= 30$) sampled uniformly amidst uncommon words (frequency $= 3$); model extracts the 10 most common words.
     - **Frequent Words Extraction (FWE):** Words sampled according to a Zeta distribution ($\alpha = 2.0$); model extracts the top-3 most frequent words.
  4. *Question Answering (2 tasks):*
     - **Single-Hop QA:** SQuAD golden paragraph buried inside randomly sampled distractor paragraphs.
     - **Multi-Hop QA:** HotpotQA golden paragraphs buried inside distractor paragraphs, requiring multi-document evidence synthesis.

- **Mathematical Formulations & Metrics:**
  - **Zeta Distribution for Frequent Words Extraction (FWE):**
    $$P(k) = \frac{k^{-\alpha}}{\zeta_N(\alpha)} = \frac{k^{-\alpha}}{\sum_{n=1}^N n^{-\alpha}}$$
    Immediately below the equation: $k$ is the frequency rank of the word in the vocabulary, $N$ is the total vocabulary size determined by the sequence length, $\alpha$ is the shape parameter (set to $\alpha = 2.0$), and $\zeta_N(\alpha)$ is the generalized harmonic number that normalizes the probability distribution.
  - **Effective Context Length:**
    $$L_{\text{eff}} = \max \{ L \in \{4\text{K}, 8\text{K}, 16\text{K}, 32\text{K}, 64\text{K}, 128\text{K}\} \mid \text{Acc}(L) \ge \tau \}$$
    Immediately below the equation: $L_{\text{eff}}$ is the maximum context length at which the model maintains acceptable performance, $\text{Acc}(L)$ is the macro-average accuracy across all 13 RULER tasks at length $L$, and $\tau = \text{Acc}_{\text{LLaMA-2-7B}}(4\text{K}) = 85.6\%$ is the qualitative threshold benchmarked from LLaMA-2-7B chat at $4\text{K}$ context.
  - **Weighted Average Scores ($\text{wAvg}$):**
    $$\text{wAvg}_{\text{inc}} = \sum_{i=1}^N w_i^{\text{inc}} \cdot \text{Acc}(L_i), \quad w_i^{\text{inc}} = \frac{i}{\sum_{j=1}^N j} = \frac{i}{21} \quad (N=6)$$
    $$\text{wAvg}_{\text{dec}} = \sum_{i=1}^N w_i^{\text{dec}} \cdot \text{Acc}(L_i), \quad w_i^{\text{dec}} = \frac{N - i + 1}{\sum_{j=1}^N j} = \frac{7 - i}{21} \quad (N=6)$$
    Immediately below the equations: $L_i \in \{4\text{K}, 8\text{K}, 16\text{K}, 32\text{K}, 64\text{K}, 128\text{K}\}$ for $i=1,\dots,6$; $\text{wAvg}_{\text{inc}}$ applies linearly increasing weights to simulate workloads dominated by long prompts, while $\text{wAvg}_{\text{dec}}$ applies linearly decreasing weights to simulate workloads where short prompts dominate.

- **Summary Benchmark Results (from Paper Table 3):**

| Model | Claimed Context | Effective Context ($L_{\text{eff}}$) | 4K | 8K | 16K | 32K | 64K | 128K | Macro Avg. | $\text{wAvg}_{\text{inc}}$ | $\text{wAvg}_{\text{dec}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaMA-2 (7B, Baseline)** | 4K | — | **85.6** | — | — | — | — | — | — | — | — |
| **Gemini-1.5-Pro** | 1M | **>128K** | 96.7 | 95.8 | 96.0 | 95.9 | 95.9 | 94.4 | **95.8** | **95.5** (1st) | **96.1** (1st) |
| **GPT-4** | 128K | **64K** | 96.6 | 96.3 | 95.2 | 93.2 | 87.0 | 81.2 | **91.6** | **89.0** (2nd) | **94.1** (2nd) |
| **LLaMA-3.1 (70B)** | 128K | **64K** | 96.5 | 95.8 | 95.4 | 94.8 | 88.4 | 66.6 | **89.6** | **85.5** (4th) | **93.7** (3rd) |
| **Qwen2 (72B)** | 128K | **32K** | 96.9 | 96.1 | 94.9 | 94.1 | 79.8 | 53.7 | **85.9** | **79.6** (9th) | **92.3** (4th) |
| **Command-R-plus (104B)** | 128K | **32K** | 95.6 | 95.2 | 94.2 | 92.0 | 84.3 | 63.1 | **87.4** | **82.7** (7th) | **92.1** (5th) |
| **GLM-4 (9B)** | 1M | **64K** | 94.7 | 92.8 | 92.1 | 89.9 | 86.7 | 83.1 | **89.9** | **88.0** (3rd) | **91.7** (6th) |
| **LLaMA-3.1 (8B)** | 128K | **32K** | 95.5 | 93.8 | 91.6 | 87.4 | 84.7 | 77.0 | **88.3** | **85.4** (5th) | **91.3** (7th) |
| **Yi (34B)** | 200K | **32K** | 93.3 | 92.2 | 91.3 | 87.5 | 83.2 | 77.3 | **87.5** | **84.8** (6th) | **90.1** (10th) |
| **Phi-3-medium (14B)** | 128K | **32K** | 93.3 | 93.2 | 91.1 | 86.8 | 78.6 | 46.1 | **81.5** | **74.8** (10th) | **88.3** (11th) |
| **Mistral-v0.2 (7B)** | 32K | **16K** | 93.6 | 91.2 | 87.2 | 75.4 | 49.0 | 13.8 | **68.4** | **55.6** (13th) | **81.2** (12th) |
| **DBRX (36B/132B)** | 32K | **8K** | 95.1 | 93.8 | 83.6 | 63.1 | 2.4 | 0.0 | **56.3** | **38.0** (14th) | **74.7** (14th) |
| **LongChat (7B)** | 32K | **<4K** | 84.7 | 79.9 | 70.8 | 59.3 | 0.0 | 0.0 | **49.1** | **33.1** (16th) | **65.2** (16th) |

---

## 3. Role in Senior Project ([`docs/proposal.md`](../../proposal.md) Section 3.3.2 & 3.3.4)

- **Role:** **Diagnostic Metric — Positional Retrieval & Effective Context Verifier**
  - As defined in Proposal Section 3.3.4, RULER is not run for every routine profiling iteration. Instead, it serves as our targeted **Diagnostic Metric** to explain and isolate quality regressions flagged by our primary metrics.
  - **Specific Trigger Conditions:**
    1. *Memory Adherence Drop:* Triggered when conversational accuracy on LongMemEval drops significantly below the uncompressed baseline.
    2. *Logit Divergence Anomaly:* Triggered when token-level KL divergence or top-1 logit agreement flags substantial distortion that high-level dialogue evaluation fails to localize.
  - **Diagnostic Value:** Because RULER strictly isolates sequence length ($4\text{K}\text{--}32\text{K}$), distractor density, and needle position, it disambiguates whether a failure is caused by an eviction/offload bug (e.g., evicted key dropped or corrupted over PCIe) versus native model attention breakdown at long sequence lengths.

- **Adapting RULER for `kv_bench` on Consumer Hardware (RTX 4060 8GB / RTX 4090 16GB):**
  - **Pruned Subtask Suite:** In accordance with Proposal Section 3.3.4, we omit the synthetic word-counting aggregation subtasks (CWE and FWE) as they are unrepresentative of conversational dialogue. We retain the S-NIAH, MK-NIAH, MQ-NIAH, and Variable Tracking (VT) suites.
  - **Target Context Sweeps ($4\text{K}\text{--}32\text{K}$):** We calibrate test sweeps across $4\text{K}, 8\text{K}, 16\text{K},$ and $32\text{K}$ tokens. On an 8 GB RTX 4060, an uncompressed $32\text{K}$ BF16 KV cache crashes with Out-Of-Memory (OOM) errors during generation. This provides a direct benchmark: our co-optimized engine (priority scoring + selective 4-bit KV quantization + PCIe CPU offload) must execute $32\text{K}$ context sequences without OOM while maintaining effective retrieval accuracy above the uncompressed baseline.
  - **Compatibility with Pinned Models ([`docs/MODELS.md`](../../MODELS.md)):** Tested on our pinned dense GQA models (`Qwen 3 4B`, `Phi-4 mini`, `SmolLM3 3B`, `Qwen 3 1.7B`) in native BF16, verifying that key precision and priority scoring retain multi-needle and variable chain retrieval without weight quantization.
