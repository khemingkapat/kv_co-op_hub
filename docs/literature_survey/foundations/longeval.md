# LongEval & LongChat: How Long Can Open-Source LLMs Truly Promise on Context Length?

## 0. Plain-English TL;DR (What It Is & Why It Matters)

- **The Big Picture:**
  In mid-2023, open-source model creators began advertising massive context windows ranging from eight thousand up to eighty-four thousand words. However, the LMSYS team (the creators of Vicuna and FastChat) discovered that these claims were largely misleading: while the models could generate fluent-sounding sentences at long contexts, they suffered from catastrophic amnesia, completely failing to retrieve specific facts mentioned earlier in the conversation. To expose this gap, LMSYS created **LongEval**—a standardized benchmark testing both high-level topic recall and exact fact lookup—and released **LongChat**, demonstrating how to properly stretch a model's context from two thousand to sixteen thousand tokens by condensing position embeddings and finetuning on long conversations.

- **Core Concepts Explained Simply:**
  - **The "Fluent but Amnesic" Trap:** A language model can easily maintain proper English grammar and prose style over thousands of words while remaining completely blind to the actual information hidden in its history. Evaluating long-context capability purely by text fluency or perplexity masks severe retrieval failure.
  - **Coarse-Grained Topic Retrieval:** A test that mimics a human having an extended conversation spanning multiple subjects over time. At the very end of the chat, the model is asked what topic was discussed first. This is considered "coarse" because even if the model's memory is fuzzy by several hundred words, it can still identify the general subject.
  - **Fine-Grained Line Retrieval:** A strict lookup test where the model is given hundreds of lines of synthetic data pairs—such as unique dictionary keys mapped to specific numbers—and asked to look up the exact number for one specific key. This tests pinpoint memory precision where a single misplaced token results in complete failure.
  - **Position Embedding Condensation:** Standard Transformer models use rotational position encoders calibrated to a maximum sequence length (such as two thousand tokens). If you feed them longer text without adjustments, the internal rotation angles enter completely unfamiliar territory and the attention mechanism breaks down. By dividing the position indices so that sixteen thousand tokens fit mathematically into the original two thousand token range, the model can reuse its pretrained attention patterns across much longer documents.
  - **The Context Edge Cliff:** Even after stretching a model to sixteen thousand tokens, its retrieval accuracy drops sharply when facts are placed near the very end of its supported window. Reliable memory is always smaller than the theoretical maximum context window.

---

## 1. Key Takeaways for Our Project

- **What This Means for Our 8–16 GB Consumer GPUs:**
  - Modern open-weight models pinned for our project (such as Qwen 3 4B, Phi-4 mini, and Ministral 3 3B) ship with native support for thirty-two thousand to two hundred sixty-two thousand tokens. We do not need to perform position interpolation or finetuning ourselves because modern models natively handle large position ranges.
  - However, **LongEval's diagnostic revelation directly governs how we benchmark our system**:
    - When we compress or offload the Key-Value (KV) cache to stay within our 8 GB and 16 GB VRAM limits, standard tests like text generation or high-level summarization will make an aggressive eviction policy look artificially successful.
    - If an eviction policy aggressively drops "non-essential" tokens, the model may still produce grammatically flawless text while silently destroying fine-grained facts.
    - Therefore, our test harness (`kv_bench`) must split evaluation into coarse-grained task benchmarks (conversation flow) and fine-grained needle retrieval benchmarks (pinpoint data lookup). If our 3-axis co-optimization passes coarse retrieval but fails fine-grained line lookup, our policy is dropping critical Key-Value states.

- **Formulas & Rules of Thumb to Keep:**
  - **Effective Retrieval Window vs. Claimed Context Window:**
    As an empirical rule of thumb from LMSYS, an open model's reliable retrieval context is typically only twenty percent to fifty percent of its claimed theoretical window unless explicitly finetuned and verified with fine-grained lookup tests.
  - **Two-Tier Diagnostic Granularity:**
    Always test two distinct retrieval regimes:
    1. *Chunk-level tolerance:* Target information surrounded by four hundred to six hundred token topic blocks.
    2. *Token-level tolerance:* Exact key-value register lookup where tolerance is zero tokens.

- **Verification of Team Notes (`manual_reviews.md`):**
  - *Confirmed:* Row 21 in `manual_reviews.md` accurately identified LongEval's core contribution: slicing evaluation into coarse-grained topic retrieval and fine-grained line retrieval up to sixteen thousand tokens, and noting that position interpolation alone degrades performance without curated finetuning.
  - *Refinements & Corrections from the PDF Ground Truth:*
    - The team's note stated that models struggle when retrieving facts near the beginning or middle. The LongEval report specifically highlights that models suffer an **accuracy drop near the maximal context boundary** (the 12K–16K boundary on a 16K model) due to fine-tuning length saturation, in addition to catastrophic failure on early tokens seen in unadapted baselines like vanilla LLaMA.
    - The note mentioned "statistical method" for fine-tuning; the exact method is standard next-token cross-entropy loss applied to curated multi-turn user conversations using PyTorch FSDP and FlashAttention, costing approximately 300 to 700 dollars on cloud A100 GPUs.

---

## 2. System & Architectural Analysis

- **Core Contribution:**
  The LMSYS technical report (*How Long Can Open-Source LLMs Truly Promise on Context Length?*, June 29, 2023) established the first empirical falsification of early open-source long-context claims. It introduced:
  1. **LongEval:** A reproducible diagnostic suite isolating long-context retrieval into coarse-grained (Topic Retrieval) and fine-grained (Line Retrieval) regimes while disentangling instruction-following failures.
  2. **LongChat-7B/13B:** An open-source recipe proving that Rotary Position Embedding (RoPE) linear interpolation combined with supervised finetuning on truncated long-context conversations (16K tokens) matches proprietary systems (GPT-3.5-turbo-16K and Claude-1.3-100K) up to 12K tokens.

- **Architectural Bottlenecks / Failure Modes Addressed:**

  ### 1. Rotary Position Embedding (RoPE) Linear Condensation
  Pretrained Transformer architectures like LLaMA-1 compute attention using Rotary Position Embeddings ($\text{RoPE}$). Given a token at integer sequence position $p \in \mathbb{N}$, query state $\mathbf{q}$, and key state $\mathbf{k}$, the rotary transformation applies a block-diagonal rotation matrix:
  $$\mathbf{q}_p = \mathbf{R}_{\Theta, p}^d \mathbf{q}, \quad \mathbf{k}_p = \mathbf{R}_{\Theta, p}^d \mathbf{k}$$
  where $\mathbf{R}_{\Theta, p}^d$ rotates orthogonal two-dimensional chunks of the embedding vector by angle $p \theta_i$ with frequency base $\theta_i = 10000^{-2(i-1)/d}$.

  When feeding sequences longer than the pretrained context length ($L_{\text{train}} = 2048$), unseen position indices $p > L_{\text{train}}$ yield out-of-distribution rotation angles, leading to catastrophic attention logit explosions ($\mathbf{q}_p^T \mathbf{k}_m \gg 0$) and generation collapse.

  LongChat introduces linear position condensation by defining a condensation ratio $r$:
  $$r = \frac{L_{\text{target}}}{L_{\text{train}}} = \frac{16384}{2048} = 8$$

  The modified position index $p'$ supplied to the positional transformation is scaled linearly:
  $$p' = \frac{p}{r}$$

  $$\mathbf{q}_p = \mathbf{R}_{\Theta, \frac{p}{r}}^d \mathbf{q}, \quad \mathbf{k}_p = \mathbf{R}_{\Theta, \frac{p}{r}}^d \mathbf{k}$$
  *Plain English:* Every token's position index $p$ is divided by the scaling factor $r=8$. For example, token $10{,}000$ is mapped to position index $1250$, and its adjacent neighbor $10{,}001$ is mapped to $1250.125$. This compresses the entire $16\text{K}$ sequence into the familiar $[0, 2048]$ angular domain that the model learned during pretraining, requiring zero extra model parameters.

  ### 2. Angular Resolution Shrinkage & Finetuning Necessity
  While position condensation prevents out-of-distribution rotation angles, it divides the angular distance between neighboring tokens by $r$:
  $$\Delta \theta = \frac{\theta_i}{r}$$
  *Plain English:* Because neighboring tokens are rotated by much smaller angular increments, the model's attention heads can no longer distinguish adjacent tokens clearly, causing high perplexity and severe retrieval degradation if applied without training. LMSYS demonstrated that full-parameter supervised finetuning on multi-turn conversations ($80\text{K}$ dialogs for 7B, $18\text{K}$ dialogs for 13B) using standard cross-entropy loss is mandatory to restore token discriminability.

  ### 3. Empirical LongEval Results Across Models
  The report evaluated six representative models across varying context scales:

  #### Table 1: Model Specifications & Claimed vs. Actual Architectures
  | Model | Parameter Size | Instruction Tuned? | Pretrained Context | Finetuned Context | Claimed Context | Open Source? |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **MPT-30B-chat** | 30B | Yes | 8K | — | 8K | Yes |
  | **MPT-7B-storywriter** | 7B | Yes | 2K | 65K | 84K | Yes |
  | **ChatGLM2-6B** | 6B | Yes | 32K | 8K | 8K | Yes |
  | **LongChat-13B-16K** | 13B | Yes | 2K | 16K | 16K | Yes |
  | **GPT-3.5-turbo** | Proprietary | Yes | — | — | 16K | No |
  | **Claude-1.3** | Proprietary | Yes | — | — | 100K | No |

  #### Table 2: General Reasoning Capability (MT-Bench Human Preference Scores)
  | Model | MT-Bench Score (out of 10) |
  | :--- | :--- |
  | **Vicuna-13B** (Reference baseline) | 6.39 |
  | **WizardLM-13B** | 6.35 |
  | **LongChat-13B-16K** | **5.95** |
  | **Baize-v2-13B** | 5.75 |
  | **Nous-Hermes-13B** | 5.51 |
  | **Alpaca-13B** | 4.53 |

  *Finding:* LongChat-13B retained 93.1% of Vicuna-13B's short-context MT-Bench score ($5.95 \text{ vs. } 6.39$), demonstrating that context extension via linear RoPE interpolation preserves core conversational and instruction-following abilities.

  #### Table 3: Academic Long-Document Reading Comprehension (ZeroScrolls Qasper Validation Set)
  | Model | Qasper QA ($F_1$ Score) |
  | :--- | :--- |
  | **GPT-4-8K** | 0.356 |
  | **LongChat-13B-16K** | **0.286** |
  | **LongChat-7B-16K** | **0.275** |
  | **Vicuna-13B-v1.3** | 0.220 |
  | **Vicuna-7B-v1.3** | 0.190 |

  *Finding:* Extending the effective context from 2K to 16K boosted Qasper question answering $F_1$ by $+30.0\%$ on 13B ($0.286 \text{ vs. } 0.220$) and $+44.7\%$ on 7B ($0.275 \text{ vs. } 0.190$).

  #### Table 4: Qualitative Retrieval Ability Breakdown at Claimed Context Length
  | Model | Claimed Context | Natural Text Generation | Coarse Topic Retrieval | Fine-Grained Line Retrieval |
  | :--- | :--- | :--- | :--- | :--- |
  | **LongChat-13B-16K** | 16K | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
  | **MPT-30B-chat** | 8K | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
  | **MPT-7B-storywriter** | 80K | ⭐⭐⭐ | ⭐⭐ | ⭐ |
  | **ChatGLM2-6B** | 8K | ⭐⭐⭐ | ⭐⭐ | ⭐ |
  | **GPT-3.5-turbo** | 16K | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
  | **Claude-1.3** | 100K | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

  *Empirical Degradation Metrics from Report Text:*
  - **MPT-7B-storywriter (claimed 84K via ALiBi):** Achieved $<50\%$ accuracy on coarse topic retrieval at only $16\text{K}$ tokens (less than one-fifth of advertised capacity). On fine-grained line retrieval, accuracy plummeted from $\approx 50\%$ down to $\approx 30\%$.
  - **ChatGLM2-6B (claimed 8K):** Failed at coarse topic retrieval at $6\text{K}$ length ($46\%$ accuracy), and degraded to $32\%$ accuracy on fine-grained line retrieval at only $5\text{K}$ tokens.
  - **LongChat-13B-16K:** Maintained $>90\%$ retrieval accuracy matching GPT-3.5-turbo up to $12\text{K}$ tokens, but exhibited an accuracy drop between $12\text{K}\text{--}16\text{K}$ on fine-grained line lookup due to training sequence length edge effects.

- **Relevance to `kv_bench` Baseline Work Packages (`WP-1`, `WP-2`, `WP-3`):**
  - **`WP-1` (Baseline Characterization & Model Pinning):**
    As documented in [`docs/MODELS.md`](../../MODELS.md), modern models pinned for our benchmark (`Qwen 3 4B`, `Phi-4 mini`, `Ministral 3 3B`, `SmolLM3 3B`, `Qwen 3 1.7B`) feature native context windows ranging from $32\text{K}$ up to $262\text{K}$ tokens without requiring manual RoPE linear scaling. LongEval provides the theoretical baseline explaining why older models (LLaMA-1, Vicuna) suffered from the context barrier, and sets the foundation for modern RoPE frequency scaling ($\theta_{\text{base}} = 10{,}000 \rightarrow 10{,}000{,}000$).
  - **`WP-2` (Diagnostic Workload Profile):**
    LongEval directly inspired modern diagnostic suites (Needle In A Haystack, RULER, and LongMemEval). For our senior project profiling harness (`kv_bench`), we implement two distinct diagnostic suites:
    1. *Coarse Granularity (Topic / Turn Retrieval):* Tests whether hierarchical caching or episode clustering (e.g., EpiCache) preserves high-level conversation structure.
    2. *Fine Granularity (Key-Value Line / Variable Tracking):* Tests whether priority-based eviction or INT4/INT8 selective quantization destroys exact token registers.
  - **`WP-3` (Trilemma Co-Optimization & Falsification Guard):**
    LongEval proves that language models can maintain high generative fluency (low language modeling perplexity) while failing catastrophic amnesia tests. In our 3-axis co-optimization experiments:
    - We must **never** declare a KV cache compression or eviction scheme successful based solely on perplexity or conversational fluency.
    - We must run both coarse topic retrieval and fine-grained line lookup across sequence depths ($0\%, 25\%, 50\%, 75\%, 100\%$) and lengths ($4\text{K} \rightarrow 32\text{K}$) to detect whether our policy evicts needle tokens prematurely.
