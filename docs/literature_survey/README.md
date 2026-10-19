# Literature Survey & Technical Knowledge Base

> **Senior Project:** Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs  
> **Master Roadmap:** [`WORK_PACKAGES.md`](../../WORK_PACKAGES.md) (`WP-3`, `WP-4`)  
> **Execution Playbook & Skill:** [`.agents/skills/literature-review/SKILL.md`](../../.agents/skills/literature-review/SKILL.md)  
> **Team Notes:** [`manual_reviews.md`](../../manual_reviews.md)  
> **Local PDF Staging:** [`papers/`](../../papers/) *(gitignored)*

---

## 📚 Survey Overview & Master Index

This directory maintains the implementation-grade literature base supporting the Senior Project proposal, runtime engine design in `kv_bench`, and final thesis synthesis.

### Tier 1: Systems & Architecture Foundations (`foundations/`)
Foundational systems literature detailing the hardware memory wall, prefill vs. decode characteristics, numerical equivalence, and serving architectures.

| Topic / System | Primary Focus | Local Source / PDF | Review File | Status |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA Inference Optimization** | Prefill vs. Decode dynamics, KV cache formula | `papers/Mastering LLM Techniques_ Inference Optimization...pdf` | [`foundations/nvidia_inference_optimization.md`](foundations/nvidia_inference_optimization.md) | 🟢 Complete |
| **The Illusion of Equivalence (Chodavarapu 2026)** | FP16 non-associativity & KV cache numerical divergence | `papers/2604.15409v1.pdf` | `foundations/illusion_of_equivalence.md` | 🟡 Staged (In manual_reviews) |
| **LongEval / LMSYS Blog (2023)** | Context window extension & fine-tuning evaluation | `papers/How Long Can Open-Source LLMs Truly Promise...pdf` | `foundations/longeval.md` | 🟡 Staged (In manual_reviews) |
| **FlashAttention (1, 2, 3)** | IO-Aware Attention Tiling (SRAM vs. HBM) | *(Web / ArXiv)* | `foundations/flashattention.md` | ⚪ Planned |
| **PagedAttention / vLLM** | Virtual Memory Paging & elimination of fragmentation | *(Web / SOSP 2023)* | `foundations/pagedattention_vllm.md` | ⚪ Planned |
| **KV Cache Survey (Jiang 2026)** | 3-Axis Taxonomy & Trilemma framework | `papers/` (Survey [21]) | `foundations/kv_survey_jiang2026.md` | ⚪ Planned |

---

### Tier 2: KV Cache Optimization Techniques (`techniques/`)
In-depth technical dissections of specific eviction, quantization, and offloading mechanisms evaluated for reproduction or component inspiration.

| Technique | Primary Axis | Local Source / PDF | Review File | Status |
| :--- | :--- | :--- | :--- | :--- |
| **EpiCache (ICML 2026)** | Structural + Spatial | `papers/2509.17396v4.pdf` | [`techniques/epicache.md`](techniques/epicache.md) | 🟢 Complete |
| **FlexGen (ICML 2023)** | Spatial + Structural (Offloading) | `papers/2303.06865v2.pdf` | `techniques/flexgen.md` | 🟡 Staged (In manual_reviews) |
| **H2O (NeurIPS 2023)** | Structural (Eviction / Heavy Hitters) | `papers/2306.14048v3.pdf` | `techniques/h2o.md` | 🟡 Staged (In manual_reviews) |
| **StreamingLLM (ICLR 2024)** | Structural (Attention Sinks + Window) | `papers/2309.17453v4.pdf` | `techniques/streamingllm.md` | 🟡 Staged (In manual_reviews) |
| **CacheGen (SIGCOMM 2024)** | Structural + Spatial (Streaming / Codec) | `papers/2310.07240v6.pdf` | `techniques/cachegen.md` | 🟡 Staged (In manual_reviews) |
| **QServe / QoQ (MLSys 2025)** | Structural (W4A8KV4 + Kernel Fusion) | `papers/2405.04532v3.pdf` | `techniques/qserve.md` | 🟡 Staged (In manual_reviews) |
| **InfiniGen (OSDI 2024)** | Spatial + Temporal (Prefetching) | `papers/2406.19707v1.pdf` | `techniques/infinigen.md` | 🟡 Staged (In manual_reviews) |
| **ShadowKV (2024)** | Structural + Spatial (Key GPU / Val CPU) | `papers/2410.21465v3.pdf` | `techniques/shadowkv.md` | 🟡 Staged (In manual_reviews) |
| **ArkVale (NeurIPS 2024)** | Spatial + Temporal (Eviction + Recall) | `papers/NeurIPS-2024-arkvale-...pdf` | `techniques/arkvale.md` | 🟡 Staged (In manual_reviews) |
| **HeadQ (2026)** | Structural (2-bit Key Correction) | `papers/2605.03562v2.pdf` | `techniques/headq.md` | 🟡 Staged (In manual_reviews) |
| **TurboQuant / D'Alberto (2026)** | Structural (Quantization & K-V Asymmetry) | `papers/2605.08114v1.pdf` | `techniques/turboquant.md` | 🟡 Staged (In manual_reviews) |
| **SnapKV (2024)** | Structural (Clustering in Window) | *(Web / ArXiv)* | `techniques/snapkv.md` | ⚪ Planned |
| **PyramidKV (2024)** | Structural (Pyramidal Layer Budget) | *(Web / ArXiv)* | `techniques/pyramidkv.md` | ⚪ Planned |
| **KIVI (ICML 2024)** | Structural (2-bit Per-Channel/Token) | *(Web / ArXiv)* | `techniques/kivi.md` | ⚪ Planned |

---

### Tier 3: Benchmarks, Datasets & Evaluation Metrics (`benchmarks/`)
Workload specifications, evaluation harnesses, and failure mode diagnostics mapping systems trade-offs to conversational quality.

| Benchmark / Metric | Category | Local Source / PDF | Review File | Status |
| :--- | :--- | :--- | :--- | :--- |
| **LongMemEval (ICLR 2025)** | Task Quality Check (Multi-Session QA) | `papers/2410.10813v2.pdf` | [`benchmarks/longmemeval.md`](benchmarks/longmemeval.md) | 🟢 Complete |
| **LoCoMo (ACL 2024)** | Long-Term Conversational QA & Timeline | `papers/2402.17753v1.pdf` | `benchmarks/locomo.md` | 🟡 Staged (In manual_reviews) |
| **RULER (COLM 2024)** | Synthetic Retrieval & Real Context Size | `papers/2404.06654v3.pdf` | `benchmarks/ruler.md` | 🟡 Staged (In manual_reviews) |
| **Memory NIAH (IBM 2024)** | Positional Retrieval on Memory LLMs | `papers/2407.01437v2.pdf` | `benchmarks/memory_niah.md` | 🟡 Staged (In manual_reviews) |
| **SEAL (2025)** | Attention Scaling for Long-Context Recall | `papers/2501.15225v1.pdf` | `benchmarks/seal.md` | 🟡 Staged (In manual_reviews) |
| **RealTalk (2025)** | 21-Day Real-World Messaging Dataset | `papers/2502.13270v1.pdf` | `benchmarks/realtalk.md` | 🟡 Staged (In manual_reviews) |
| **PG-19 / WikiText-2** | Perplexity Delta Baseline Check | *(Workload Staging)* | `benchmarks/pg19_wikitext2.md` | ⚪ Planned |
