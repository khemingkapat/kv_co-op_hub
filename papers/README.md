# Local Paper PDF Staging Directory (`papers/`)

Drop downloaded research paper PDFs into this directory (either directly in `papers/` or under optional subfolders). All `*.pdf` files in `papers/` are ignored by Git via [`.gitignore`](../.gitignore).

## Recommended Naming Convention (Optional, for Easy Matching)

You can name PDFs however you like, or use short paper keys matching the target review filenames:

### Tier 1: Foundations (`docs/literature_survey/foundations/`)
- `flashattention.pdf` / `flashattention2.pdf` / `flashattention3.pdf`
- `pagedattention_vllm.pdf` — Kwon et al., SOSP 2023
- `kv_survey_jiang2026.pdf` — Jiang et al., ACL Findings 2026 (*Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization*)
- `kv_survey_zheng2024.pdf` — Zheng et al., 2024

### Tier 2: KV Cache Optimization Techniques (`docs/literature_survey/techniques/`)
- **Eviction & Pruning:**
  - `h2o.pdf` — Zhang et al., NeurIPS 2023
  - `streamingllm.pdf` — Xiao et al., ICLR 2024
  - `snapkv.pdf` — Li et al., 2024
  - `pyramidkv.pdf` — Cai et al., 2024
  - `fastgen.pdf` — Ge et al., 2023
- **Quantization & Compression:**
  - `qserve.pdf` — Lin et al., MLSys 2025
  - `headq.pdf` — Ruiz Williams, 2026
  - `cachegen.pdf` — Liu et al., SIGCOMM 2024
  - `kivi.pdf` — Liu et al., ICML 2024
- **Spatial & Temporal (Offloading & Prefetching):**
  - `flexgen.pdf` — Sheng et al., ICML 2023
  - `shadowkv.pdf` — Sun et al., 2024
  - `infinigen.pdf` — Lee et al., OSDI 2024
  - `arkvale.pdf` — Chen et al., NeurIPS 2024
  - `epicache.pdf` — Kim et al., ICML 2026

### Tier 3: Benchmarks & Metrics (`docs/literature_survey/benchmarks/`)
- `longmemeval.pdf` — Wu et al., ICLR 2025
- `ruler.pdf` — Hsieh et al., 2024
- `babilong.pdf` — Kuratov et al., 2024
