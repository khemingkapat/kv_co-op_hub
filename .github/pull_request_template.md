## 📌 Summary & Context
- **Work Package Tag:** `[e.g., WP-1, WP-2, WP-3]`
- **Contribution Type:** `[ 📊 Data/Manifest | 📄 Research/Docs | 🎓 Academic Deliverable | ⚙️ Tooling ]`
- **Connected Parent Issue:** `Part of khemingkapat/kv_co-op_hub#<ISSUE_NUMBER>`

---

## 🛡️ Scientific Scope & Integrity Gatekeeping (Mandatory)
*Every PR must satisfy these criteria before merging:*

- [ ] **1. Hardware Capacity Specification:**
  - Evaluated workloads fit strictly within target consumer GPU budgets (RTX 4090 cluster node capped at 16.00 GiB VRAM, or RTX 4060 local testbed at 8.00 GiB VRAM).
- [ ] **2. Co-Optimization Axis Classification:**
  - Explicitly categorizes the contribution under its corresponding axis or multi-axis interaction:
    - [ ] `STRUCTURAL` (KV-Cache Quantization FP8/INT4, Eviction, Windowing)
    - [ ] `SPATIAL` (CPU Host RAM Swapping, Tiered Memory Offloading)
    - [ ] `TEMPORAL` (Prefix Caching, Prefetching, Async Transfer Pipelining)
    - [ ] `CO-DESIGN` (Multi-Axis Coordination Policy)
    - [ ] `METHODOLOGY / GOVERNANCE` (Specs, literature surveys, manifests, or deliverables)
- [ ] **3. Controlled Base Model & Clean Repository:**
  - Uses pinned base model checkpoints from [`docs/MODELS.md`](docs/MODELS.md) (KV-cache compression/quantization belongs under `STRUCTURAL`, whereas base model weights remain matched to the pinned reference unless explicitly evaluating a joint weight-KV kernel).
  - Zero heavy binary checkpoints (`.pt`, `.bin`, `.safetensors`, `.gguf`), raw Slurm `.log` files, or unlicensed PDFs committed.
  - Per-token raw logprob references (`*_ref.json`) are restricted strictly to `results/baselines/references/`; candidate experimental sweeps store only computed summary metrics (`top1_agreement`, `kl_div_mean`, `kl_div_p99`).
- [ ] **4. Milestone & Scope Discipline:**
  - Changes are strictly bounded to the active Work Package milestone without premature architectural implementations outside current scope.
- [ ] **5. Provenance & Reproducibility Recorded:**
  - All benchmark numbers, datasets, or references include a companion manifest in `results/manifests/` recording Slurm Job ID, cluster node, environment versions, upstream `kv_bench` commit SHA, and SHA256 checksums.

---

## 🧪 Verification & Evidence
*Describe what local checks or verification commands were run:*
- Command / Test: `...`
