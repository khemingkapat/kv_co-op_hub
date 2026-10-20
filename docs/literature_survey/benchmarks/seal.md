# SEAL: Scaling to Emphasize Attention for Long-Context Retrieval

## 0. Plain-English TL;DR (What It Tests & Why We Use It)

- **What This Benchmark & Diagnostic Is:**
  SEAL is a diagnostic study and lightweight attention-scaling framework that investigates why large language models struggle to recall information when reading long documents. As context grows, models often suffer from a lost-in-the-middle effect where retrieval accuracy plummets. The authors discovered that transformer attention heads do not contribute equally to retrieval: a small subset of heads acts as dedicated retrieval specialists, while other heads actively compete with them by generating noisy signals. SEAL proves that selectively boosting the helpful heads and dampening the noisy heads restores retrieval accuracy across long contexts without full parameter fine-tuning.

- **Why Standard Tests Aren't Enough:**
  - Simple language predictability checks like Perplexity only measure whether text flows naturally, completely missing whether the model lost a critical needle hidden deep in the prompt.
  - Standard Needle-In-A-Haystack tests treat the entire neural network as a black box, reporting only a pass or fail score without showing which internal components failed or why recall collapsed.
  - SEAL evaluates retrieval across both token length and depth distributions (from zero percent at the very beginning to one hundred percent at the very end of the text), revealing exactly where and why retrieval breaks down.

- **How It Works Step-by-Step:**
  1. *Synthetic Context-Format Generation:* Instead of gathering expensive human-annotated data, SEAL creates roughly 50 synthetic test samples matching the format of the target task (for example, finding a 5-digit number associated with a line key, or retrieving a simple fact from a book excerpt).
  2. *Sensitivity Probing and Multiplier Training:* The model runs a single training pass over these 50 synthetic samples to learn a tiny multiplier for each attention head (SEAL-H) or each hidden channel (SEAL-C). Helpful heads are scaled up, while noisy or distracting heads are scaled down.
  3. *Offline Zero-Cost Folding:* The learned multipliers are multiplied directly into the existing linear projection matrices offline before running inference. This means the model gains enhanced retrieval accuracy with zero extra math, zero added latency, and zero additional memory during text generation.

---

## 1. Key Takeaways & How We Use It in `kv_bench`

- **Specific KV-Cache Failures Exposed:**
  - **Positional Blind Spots (Lost-in-the-Middle):** Unoptimized models experience severe accuracy drops when evidence is placed between twenty-five percent and seventy-five percent context depth. In our project, if an eviction policy naively drops middle tokens or relies on simple sliding windows, it severely exacerbates this positional blindness.
  - **The Flaw of Uniform Head Eviction:** Standard inference engines allocate identical memory buffers and eviction policies across all attention heads. SEAL proves that attention heads have radically different retrieval importance. Evicting tokens from a dedicated retrieval head causes catastrophic recall failure, whereas evicting tokens from a distracting head can actually improve model accuracy.
  - **Distractor Amplification:** Later feed-forward layers can latch onto incorrect token activations produced by noisy attention heads. If key-value cache compression or quantization introduces noise into key vectors, distractor tokens drown out the correct answer.

- **Key Findings from the Paper Worth Knowing:**
  - 1. **Pruning Distracting Heads Improves Accuracy:** On LongChat at long sequence lengths, pruning individual noisy heads actually increased retrieval accuracy by over twenty percent, confirming that certain heads actively degrade long-context recall.
  - 2. **Ultra-Minimal Parameter Scaling Rescues Retrieval:** Adjusting just 1,024 head numbers (representing approximately fifteen millionths of a percent of model parameters) restored LongChat-7B retrieval accuracy from 32 percent back up to 80 percent at maximum context length.
  - 3. **Rescuing Training-Free Context Extension:** Training-free context stretch methods like NTK RoPE scaling frequently collapse to zero percent accuracy beyond their baseline window. Combining them with attention scaling restored retrieval accuracy to nearly ninety percent without heavy fine-tuning.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 20 in `manual_reviews.md` correctly identified that SEAL focuses on attention scaling across context depth distributions (0%, 25%, 50%, 75%, 100%) and adjusts an ultra-minimal parameter set rather than full model weights.
  - *Refinement from Document:* While the team notes classified SEAL primarily as a "Dataset", the paper is fundamentally an architectural diagnostic and attention-weight scaling technique using only 50 synthetic format-matching samples. For our project, its core value lies in proving attention head heterogeneity: we should not treat all key-value heads equally when making eviction and quantization decisions.

---

## 2. Workload & Diagnostic Profile

- **Publication & Authors:** Changhun Lee, Jungyu Jin, Younghyun Cho, Eunhyeok Park (POSTECH - Pohang University of Science and Technology). Published on arXiv (arXiv:2501.15225v1, Jan 2025).
- **Target Workload & Context Lengths:**
  - Models evaluated: LongChat-7B-v1.5-32K, Mistral-7B-Instruct-v0.2 (32K native window), Vicuna-7B-v1.5-16K, LongChat-13B-16K, Vicuna-13B-v1.5-16K (16K native window), and Llama-2-7b-Chat (4K base extended to 16K–18.4K).
  - Context length sweeps: 9K to 31K tokens for 32K models; 5K to 16K tokens for 16K models; extended evaluations up to 80K tokens for Mistral.
  - Depth tiers evaluated: $0\%, 25\%, 50\%, 75\%, 100\%$ depth of needle insertion.

- **Diagnostic Architecture & Attention Scaling Formulations:**
  - *Direct Effect Formulation:*
    To quantify how individual transformer components influence output token logits, the direct effect $\Delta$ is defined as:
    $$\Delta = W_{\text{head}} f(p)$$
    *Where $p$ is the input prompt, $f(p)$ is the hidden state output produced by an individual attention head or feed-forward layer, $W_{\text{head}}$ is the unembedding or downstream projection weight, and $\Delta$ represents the unnormalized scalar contribution to the final logit before the softmax function.*

  - *Head-Wise Attention Scaling (SEAL-H):*
    SEAL-H introduces a single learnable scalar $s_h \in \mathbb{R}$ for each attention head $h \in \{1, \dots, H\}$ across all layers $l \in \{1, \dots, L\}$:
    $$O_h = s_h \cdot \text{Attention}(Q_h, K_h, V_h)$$
    *Where $Q_h, K_h, V_h$ are the Query, Key, and Value projections for head $h$, $\text{Attention}(\cdot)$ is standard scaled dot-product attention, $s_h$ is the learned scalar scale factor, and $O_h$ is the scaled head output tensor.*
    - **Total Learnable Parameters:** $N_{\text{SEAL-H}} = L \times H$. For a 7B LLaMA-based model ($L=32, H=32$), $N_{\text{SEAL-H}} = 1,024$ parameters ($\approx 1.5 \times 10^{-5}\%$ of total model weights).
    - **Offline Zero-Overhead Weight Folding:** Because $s_h$ is a scalar applied to the output of head $h$, it can be absorbed offline into the output projection matrix $W_o$ or the value projection matrix $W_v$:
      $$W_{o}^{(h)} \leftarrow s_h \cdot W_{o}^{(h)} \quad \text{or} \quad W_{v}^{(h)} \leftarrow s_h \cdot W_{v}^{(h)}$$
      *Where $W_{o}^{(h)}$ is the slice of the attention output projection corresponding to head $h$, ensuring zero runtime latency overhead and zero added memory during decode.*

  - *Channel-Wise Attention Scaling (SEAL-C):*
    SEAL-C provides finer-grained control by applying a learnable vector $\mathbf{s}_h \in \mathbb{R}^{d_{\text{head}}}$ across the channel dimension $c \in \{1, \dots, d_{\text{head}}\}$:
    $$O_{h, c} = s_{h, c} \cdot \left(\text{Attention}(Q_h, K_h, V_h)\right)_c$$
    *Where $d_{\text{head}}$ is the hidden dimension per head (e.g., 128), $s_{h, c}$ is the scale factor for channel $c$ of head $h$, and $O_{h, c}$ is the resulting channel activation.*
    - **Total Learnable Parameters:** $N_{\text{SEAL-C}} = L \times H \times d_{\text{head}} = L \times d_{\text{model}}$. For a 7B model ($32 \times 4096$), $N_{\text{SEAL-C}} = 131,072$ parameters ($\approx 1.9 \times 10^{-3}\%$ of total weights). Like SEAL-H, SEAL-C folds directly into $W_o$ offline.

  - *PEFT Parameter Comparison:*
    - SEAL-H: $1,024$ parameters ($\approx 0.000015\%$).
    - SEAL-C: $131,072$ parameters ($\approx 0.0019\%$).
    - SEAL-L (LoRA rank $r=4$ on QKVO): $4.2\text{M}$ parameters ($\approx 0.062\%$, $\approx 40\times$ larger than SEAL-C and $4,000\times$ larger than SEAL-H).
    - SEAL-D (DoRA rank $r=4$ on QKVO): $4.7\text{M}$ parameters ($\approx 0.070\%$).

- **Empirical Results from Paper Tables:**

  *Table 1: LongEval Line Retrieval Accuracy Across Context Lengths (0.00 to 1.00 score)*
  | Model | Method | Tunable Params (#, %) | 9K | 14K | 19K | 23K | 28K | 31K |
  | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **LongChat-7B-v1.5-32K** | Baseline | 0 (0.0%) | 0.98 | 0.96 | 0.84 | 0.54 | 0.38 | 0.32 |
  | | SEAL-H | 1.0K (1.5e-5%) | 1.00 | 1.00 | 0.98 | 1.00 | 0.94 | 0.80 |
  | | SEAL-C | 131.1K (1.9e-3%) | 0.98 | 0.96 | 0.94 | 0.92 | 0.94 | 0.88 |
  | | SEAL-L (LoRA) | 4.2M (6.2e-2%) | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.80 |
  | | SEAL-D (DoRA) | 4.7M (7.0e-2%) | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.86 |
  | **Mistral-7B-Instruct-v0.2** | Baseline | 0 (0.0%) | 0.98 | 1.00 | 0.90 | 0.86 | 0.88 | 0.94 |
  | | SEAL-H | 1.0K (1.4e-5%) | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 |
  | | SEAL-C | 131.1K (1.8e-3%) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
  | | SEAL-L (LoRA) | 4.2M (5.8e-2%) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
  | | SEAL-D (DoRA) | 4.7M (6.5e-2%) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

  *Table 1 (Continued): 16K Models (Tested at 5K, 7K, 9K, 12K, 14K, 16K)*
  | Model | Method | Tunable Params | 5K | 7K | 9K | 12K | 14K | 16K |
  | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Vicuna-7B-v1.5-16K** | Baseline | 0 | 1.00 | 1.00 | 0.96 | 0.92 | 0.60 | 0.64 |
  | | SEAL-H | 1.0K | 1.00 | 1.00 | 1.00 | 0.98 | 0.92 | 0.84 |
  | | SEAL-C | 131.1K | 1.00 | 1.00 | 1.00 | 0.94 | 0.96 | 0.98 |
  | **LongChat-13B-16K** | Baseline | 0 | 0.96 | 0.94 | 0.92 | 0.92 | 0.80 | 0.60 |
  | | SEAL-H | 1.6K | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.92 |
  | | SEAL-C | 207.7K | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |
  | **Vicuna-13B-v1.5-16K** | Baseline | 0 | 0.98 | 0.98 | 0.94 | 0.88 | 0.68 | 0.42 |
  | | SEAL-H | 1.6K | 1.00 | 1.00 | 0.96 | 1.00 | 0.96 | 0.94 |
  | | SEAL-C | 207.7K | 1.00 | 1.00 | 0.96 | 0.98 | 0.98 | 0.94 |

  *Table 2: Context Extension on Llama-2-7b-Chat (4K Base Model Extended to 16K)*
  | Method | 5K | 7K | 9K | 12K | 14K | 16K |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | NTK Baseline (Scale factor = 4) | 0.88 | 0.32 | 0.16 | 0.00 | 0.00 | 0.00 |
  | **NTK + SEAL-C** | **0.90** | **0.92** | **0.92** | **0.84** | **0.74** | **0.88** |
  | Self-Extend Baseline ($G=6, W=1024$) | 0.76 | 0.52 | 0.46 | 0.26 | 0.22 | 0.10 |
  | **Self-Extend + SEAL-C** | **0.96** | **0.96** | **0.90** | **0.84** | **0.68** | **0.56** |

  *Table 3: Out-of-Domain Task Quality & Safety Verification (LongChat-7B-v1.5-32K)*
  | Evaluation Suite | Metric / Task | Baseline | SEAL-H | SEAL-C |
  | :--- | :--- | :---: | :---: | :---: |
  | **LongBench Single-Doc QA** | In-Domain Average | 31.87 | 33.52 | **35.46** |
  | | Out-of-Domain Average | 31.87 | **33.12** | 32.96 |
  | **LongBench Multi-Doc QA** | In-Domain Average | 23.29 | 26.23 | **31.27** |
  | | Out-of-Domain Average | 23.29 | **23.63** | 23.53 |
  | **General Capability (MMLU)** | 5-Shot Accuracy (%) | **42.53** | 42.34 | 42.17 |

---

## 3. Role in Senior Project ([`docs/proposal.md`](../../proposal.md) Section 3.3.2)

- **Role:** **Diagnostic Metric & Attention Distribution Reference**
  - Used in `WP-3` (Micro-benchmarking) and `WP-7` (Final Report Quality Verification) to evaluate how KV-cache eviction and quantization policies affect positional retrieval across sequence depths ($0\%, 25\%, 50\%, 75\%, 100\%$).
  - Serves as the primary diagnostic tool to confirm that our 3-axis co-optimization does not induce or worsen lost-in-the-middle degradation.

- **Execution Strategy on 8–16 GB GPUs:**
  - **Hardware Feasibility:** Evaluation is lightweight and executes natively on our 8 GB RTX 4060 and 16 GB RTX 4090 testbeds. The evaluation suite requires only prefill and single-token decode per question, avoiding the memory growth associated with long autoregressive generation.
  - **Target Context Sweeps:** We benchmark our pinned models (`Qwen 3 4B`, `Phi-4 mini`, `Ministral 3 3B`, `SmolLM3 3B`, and `Qwen 3 1.7B`) across scaled contexts of $4\text{K}, 8\text{K}, 16\text{K},$ and $32\text{K}$ tokens.
  - **Dual Evaluation Battery:**
    1. *LongEval Line Retrieval Suite:* Padded synthetic key-value line retrieval across context lengths to test precision digit association.
    2. *Depth-Stratified Needle In A Haystack (NIAH):* Pressure-testing 5 depth intervals ($0\%, 25\%, 50\%, 75\%, 100\%$) to produce color-mapped depth-versus-length diagnostic grids.

- **Architectural & Eviction Policy Insights for KV-Cache Co-Optimization:**
  - **Unmodified BF16 Model Weights Rule:** In our project, model weights remain strictly frozen in native BF16 without parameter fine-tuning. We do not apply SEAL's offline weight scaling directly; rather, we exploit its core architectural finding: **attention heads exhibit extreme functional heterogeneity in long-context retrieval**.
  - **Non-Uniform Head Budget Allocation (HeadQ / Tiering Alignment):**
    - Standard PagedAttention in vLLM treats all key-value heads symmetrically, allocating identical block tables and eviction rules.
    - SEAL demonstrates that a minority of heads drive accurate retrieval, while others contribute noise or format tracking.
    - In `WP-4` and `WP-6`, our priority scoring engine can leverage head-level retrieval sensitivity: heads identified as critical retrieval heads retain full BF16 precision and higher token quotas in GPU VRAM, whereas heads with neutral or negative retrieval correlation can be aggressively quantized (e.g., INT4 / FP4) or evicted to host CPU memory.
  - **Grouped-Query Attention (GQA) Coherence:**
    - Our pinned models employ 4:1 or 8:1 GQA ratios, where multiple Query heads share a single Key-Value head.
    - Under SEAL's diagnostic profile, if even one Query head in a shared GQA group functions as a primary retrieval head, the underlying KV head buffer must be designated high-priority to prevent retrieval collapse.
