---
name: literature-review
description: >-
  Executes technical, implementation-grade literature reviews for KV-cache
  management, LLM inference foundations, and long-context benchmarks in
  kv_co-op_hub. Activate this skill when the user asks to review, ingest,
  summarize, or compare research papers (from local PDFs in papers/ or web/arXiv)
  and generate structured markdown reviews in docs/literature_survey/.
---

# Literature Review Skill (`kv_co-op_hub`)

This skill defines the workflow and standardized templates for producing literature reviews in `docs/literature_survey/`. Every review uses a **Two-Layer Design**:
1. **Top Half (Plain-English Intuition & Brainstorming):** Written in clear, conversational English with **zero math symbols and zero unexplained academic jargon** so any team member can grasp how the paper works and what ideas we can steal in 30 seconds.
2. **Bottom Half (Technical & Engineering Reference):** Contains the exact algorithms, equations (with every symbol explained in plain words), empirical numbers from the paper's tables, and `kv_bench` implementation feasibility on our 8–16 GB consumer GPUs.

---

## 1. Execution Workflow

### Step 1: Source Ingestion & Grounding
1. **List & Read Local PDFs First (Ground Truth):**
   - Run `ls -la papers/` (or check [`papers/`](../../../papers/)) to see all available PDFs (files may be named by raw arXiv IDs like `2410.10813v2.pdf`, `2509.17396v4.pdf`, or by article title).
   - Call `view_file` on the target PDF file. **CRITICAL TOOL RULE:** When calling `view_file` on a `.pdf` file, **DO NOT provide `StartLine` or `EndLine` arguments** (binary files always return the entire file and will error if line numbers are passed).
   - The PDF is your **ground truth**. Extract how the method works, its exact equations, architecture diagrams, and empirical tables directly from the PDF.
2. **Check [`manual_reviews.md`](../../../manual_reviews.md) for Team Context (Verify, Do NOT Trust Blindly):**
   - Check [`manual_reviews.md`](../../../manual_reviews.md) to see if the team already has skimming notes or brainstorming ideas for this paper.
   - **WARNING — HUMAN ERROR CHECK:** [`manual_reviews.md`](../../../manual_reviews.md) was written during quick manual skimming and may contain human errors, incomplete impressions, or misunderstandings. **NEVER trust [`manual_reviews.md`](../../../manual_reviews.md) solely.** Verify every point against the PDF. Preserve and expand on the team's good brainstorming angles, and gently clarify or correct any inaccuracies using the PDF ground truth.
3. **Cross-Check Official Codebase (When Needed):** If the PDF links an official GitHub repository and implementation details need verification, use `read_url_content` / `search_web` to check its codebase structure.
4. **Zero-Hallucination Rule:** Never guess numbers. If a paper does not report tail latency (P95/P99), $\Delta\text{PPL}$, or PCIe transfer volume, explicitly write `Not reported in paper (mean throughput only)`—as that absence is itself a key finding for our Trilemma analysis.

### Step 2: Writing Style Rules (CRITICAL)
- **No Unexplained Jargon in the Top Half (`Sections 0 & 1`):**
  - Do **NOT** copy-paste dense abstract sentences from the paper into the TL;DR.
  - Do **NOT** put raw LaTeX variables (like $S_{\text{medoid}}^{(e)}$ or $\sigma_\ell$) in `Section 0` or `Section 1`.
  - If a paper invents a fancy term (e.g., *"medoid patched prompt"*, *"page bounding-box digest"*, *"rehearsal prefetching"*), explain what it actually is in simple, everyday English first (e.g., *"uses the most typical conversation chunk from that topic as a 'fake question' to test which tokens the model pays attention to"*).
- **Explain Every Equation in the Bottom Half (`Section 3`):**
  - When writing a mathematical formula in `Section 3`, always add a 1–2 line plain-English translation right below it explaining what each variable means and what the formula is actually doing.

### Step 3: Classify Target Tier & Semantic Output Path
Always save the output using a **clean, lowercase semantic slug** (`<slug>.md`), **never** the raw arXiv ID filename:
- **Tier 1 — Systems & Architecture Foundations:** `docs/literature_survey/foundations/<slug>.md`
  *(e.g., `nvidia_inference_optimization.md`, `flashattention.md`, `pagedattention_vllm.md`, `kv_survey_jiang2026.md`)*
- **Tier 2 — KV Cache Optimization Techniques (Core Focus):** `docs/literature_survey/techniques/<slug>.md`
  *(e.g., `epicache.md`, `h2o.md`, `streamingllm.md`, `snapkv.md`, `pyramidkv.md`, `fastgen.md`, `qserve.md`, `headq.md`, `cachegen.md`, `kivi.md`, `flexgen.md`, `shadowkv.md`, `infinigen.md`, `arkvale.md`)*
- **Tier 3 — Benchmarks, Datasets & Evaluation Metrics:** `docs/literature_survey/benchmarks/<slug>.md`
  *(e.g., `longmemeval.md`, `ruler.md`, `niah.md`, `babilong.md`, `pg19_wikitext2.md`)*

---

## 2. Standardized Review Templates

### A. Tier 2 Template: KV Cache Optimization Techniques (MANDATORY IN-DEPTH)

Save to: `docs/literature_survey/techniques/<technique_name>.md`

```markdown
# [Technique Name]: [Full Paper Title]

## 0. Plain-English TL;DR (What It Does & Why We Care)
- **The Core Problem:** (In 2–3 plain-English sentences, what specific problem breaks existing systems? No jargon.)
- **How It Solves It (Step-by-Step in Plain Words):**
  1. **Before / During Prefill:** (What does it do when reading the long prompt/history? Explain any fancy paper terms in simple words.)
  2. **Where the Data Lives (Storage & Compression):** (What stays on the GPU vs. what gets compressed, dropped, or moved to CPU RAM?)
  3. **During Decode (Generating New Tokens):** (When a new question/token comes in, how does the GPU use the cache to answer?)
- **How It Compares to Our Project:** (In 2–3 sentences, how is this similar to or different from our 3-axis co-optimization goal of priority scoring + selective quantization + CPU offload?)
- **The Bottom-Line Trade-off:**
  - **What You Gain:** (Plain summary of memory saved, speedups, and accuracy kept)
  - **What You Pay:** (Plain summary of the catch—extra compute, PCIe stalls, permanent data loss, or extra models needed)

## 1. Key Takeaways & Ideas We Can Reuse (Brainstorming)
- **Observations & Tricks Worth Stealing:**
  - (List 2–4 concrete, clever insights or mechanisms from the paper that we could borrow or adapt for our own policy engine, explained clearly)
- **The Catch / Blind Spots on Consumer Hardware:**
  - (What did the authors test on—e.g., A100/H100—and where might this struggle on our 8 GB RTX 4060 / 16 GB RTX 4090 over standard PCIe?)
- **Verification of Team Notes (`manual_reviews.md`):**
  - (If this paper was in `manual_reviews.md`, briefly confirm what our manual notes got right and clarify/correct any details based on the full PDF)

---

## 2. Metadata & Provenance
- **Authors & Affiliations:**
- **Publication / Venue & Year:** (e.g., NeurIPS 2024, OSDI 2024, ICML 2026, arXiv:XXXX.XXXXX)
- **Primary Axis:** [Structural | Spatial | Temporal]
- **Secondary Interacting Axis:** [None | Structural | Spatial | Temporal]
- **Official Codebase / Repository:** [URL or "None"]
- **Baseline Implementation Stack:** (e.g., HuggingFace Transformers, vLLM fork, custom CUDA/Triton kernels, and which models they tested on)

## 3. How the Algorithm Actually Works (Under the Hood)
- **Data Structures Added / Modified:** (What buffers, tables, centroids, or memory pools are added?)
- **Scoring, Compression & Decision Logic:**
  - Write the key mathematical formulas in LaTeX, and **immediately below each formula, explain in plain words what every symbol means and how the score/decision is calculated**.
  - **Execution Cadence:** Does this run once offline, per prefill block, at every single decode token, or per conversation turn?
- **Memory Hierarchy & Placement:**
  - **Keys ($K$):** Where do they live (GPU VRAM / CPU RAM / Disk) and at what precision?
  - **Values ($V$):** Where do they live and at what precision?
  - **Granularity:** Token-level, head-level, layer-level, or page/block-level?
- **Decode-Phase Overheads:**
  - Does it stall the GPU waiting for CPU/PCIe transfers?
  - Does it require sorting, top-$k$ search, or dequantization during token generation?

## 4. Real Numbers from the Paper (The Trilemma: Memory, Speed, Quality)
- **Memory Saved (Efficiency):**
  - Exact VRAM reduction or compression ratio reported in the paper's tables (specify model and context length).
- **Speed & Latency (TTFT, TPOT, Tail Latency):**
  - Exact speedups or latency numbers (prefill time, decode time, swap time).
  - **Tail Latency (P95/P99):** State clearly if reported or `Not reported in paper (mean/total latency only)`.
- **Output Quality & Trustworthiness:**
  - Perplexity change ($\Delta\text{PPL}$) on WikiText-2 / PG-19 (if tested).
  - Long-context retrieval & multi-turn accuracy numbers (e.g., LongMemEval, RealTalk, LoCoMo, NIAH, RULER).

## 5. Project Coherence & `kv_bench` Feasibility (`WP-3` & `WP-6`)
- **Project Scope Alignment ([`docs/MODELS.md`](../../MODELS.md)):**
  - **Unmodified BF16/FP16 Weights Rule:** Does this work on the KV cache alone with unmodified BF16 model weights, or does it require quantizing model weights too?
  - **Dense GQA Compatibility:** Does it work cleanly on our pinned **Grouped-Query Attention (4:1 / 8:1 GQA)** models (`Qwen 3 4B`, `Phi-4 mini`, `SmolLM3 3B`, `Qwen 3 1.7B`)?
- **Consumer Hardware Fit (8 GB RTX 4060 / 16 GB RTX 4090 Cap + 16 GB RAM):**
  - How well does this fit our 8–16 GB VRAM cap, PCIe bandwidth, and 10–15 concurrent SME user workload?
- **Integration Path in `kv_bench`:**
  - **Engineering Difficulty:** [Low: Python wrapper | Medium: PyTorch attention hooks / CPU threads | High: Custom Triton/CUDA kernels]
  - **WP-3 (Micro-Benchmarking) & WP-6 (`vLLM` PagedAttention / `LMCache`):** How would we test or adapt this without breaking contiguous PagedAttention blocks?
- **Classification for Project:** [Core Target for Reproduction | Component Inspiration for WP-4 | Comparison Baseline | Reference Only]
```

---

### B. Tier 1 Template: Foundations (Architectural Digest)

Save to: `docs/literature_survey/foundations/<topic_or_system>.md`

```markdown
# [System / Concept Name]: [Full Title]

## 0. Plain-English TL;DR (What It Is & Why It Matters)
- **The Big Picture:** (In 2–3 plain-English sentences, what does this paper/post explain or build? No buzzwords.)
- **Core Concepts Explained Simply:**
  - (3–4 bullet points breaking down the main ideas in everyday language)

## 1. Key Takeaways for Our Project
- **What This Means for Our 8–16 GB Consumer GPUs:**
  - (Why standard optimizations like FlashAttention, GQA, or PagedAttention still hit Out-Of-Memory crashes on our hardware, and how this motivates our 3-axis KV co-optimization)
- **Formulas & Rules of Thumb to Keep:**
  - (Any key formulas—like the KV cache byte size equation—with every term explained in plain words)
- **Verification of Team Notes (`manual_reviews.md`):**
  - (Confirm and expand on what our manual notes captured)
```

---

### C. Tier 3 Template: Benchmarks & Metrics (Workload Profile)

Save to: `docs/literature_survey/benchmarks/<benchmark_name>.md`

```markdown
# [Benchmark / Metric Name]: [Full Title]

## 0. Plain-English TL;DR (What It Tests & Why We Use It)
- **What This Benchmark Is:** (In 2–3 plain sentences, how does this test work? What kind of data is it?)
- **Why Standard Tests Aren't Enough:** (What real-world failure does this catch that simple perplexity or basic single-needle tests miss?)
- **How It Works Step-by-Step:** (How a test sample is built and how the model is scored, in plain words)

## 1. Key Takeaways & How We Use It in `kv_bench`
- **Specific KV-Cache Failures Exposed:**
  - (e.g., what happens if our eviction drops an early turn, or if compression blurs timestamps?)
- **Key Findings from the Paper Worth Knowing:**
  - (2–3 interesting findings from the benchmark paper itself—e.g., why long-context models still drop 30–60% accuracy, or why round-level chunking beats session-level)
- **Dataset Size, Cost & Our Project Role ([`docs/proposal.md`](../../proposal.md) Section 3.3):**
  - **Role:** [Final Report Metric | Reproduction-Only Check | Diagnostic Metric]
  - **How We Run It on 8–16 GB GPUs:** (Context lengths, subset strategy, and scoring method)
- **Verification of Team Notes (`manual_reviews.md`):**
  - (Confirm and refine what our manual notes captured against the PDF)
```
