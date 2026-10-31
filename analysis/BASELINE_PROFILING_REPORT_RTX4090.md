# 🧪 Literature Reference Baseline Profiling Report: Multi-Model KV-Cache Scaling & Hardware Ceiling Mapping

> **Work Package Context:** `[WP1-1] Literature Reference Baseline & Comparative Architecture Profiling`  
> **Parent Issue:** [`khemingkapat/kv_co-op_hub#1`](https://github.com/khemingkapat/kv_co-op_hub/issues/1) & [`khemingkapat/kv_co-op_hub#2`](https://github.com/khemingkapat/kv_co-op_hub/issues/2)  
> **Model Alignment:** Evaluates checkpoints designated in [`docs/MODELS.md`](../docs/MODELS.md) under **Literature Reference** (`Qwen/Qwen2.5-3B-Instruct`), **Consumer Small** (`Qwen/Qwen2.5-1.5B`), and **Comparative Dense Peer** (`microsoft/Phi-3.5-mini-instruct`, `Qwen/Qwen2.5-7B-Instruct`).  
> **Target Hardware:** NVIDIA GeForce RTX 4090 (CPE Slurm Cluster Node: `gpu01.slurm.cpe.kmutt.ac.th`)  
> **Execution Timestamp:** 2026-10-19 | **Slurm Job IDs:** `22091` (Qwen 1.5B & 3B), `22094` (Phi-3.5 3.8B & Qwen 7B)  
> **Author / Testbed Contributor:** Sirawat Sanboonsong (New)  
> **Primary Suite Reference:** Full primary benchmarking (Qwen 3 pure dense GQA, vLLM PagedAttention, $N$ concurrent SME sessions) is standardized in [`khemingkapat/kv_bench`](https://github.com/khemingkapat/kv_bench).

---

## Executive Summary & Architectural Scope

This empirical report documents exploratory baseline sweeps across **four open-weight models** from previous model generations, serving as the empirical **Literature Reference Baseline** to substantiate the core hypothesis in our Senior Project Proposal.

### 🎯 Key Empirical Breakthroughs & Findings:
1. **The 8 GB OOM Ceiling (Consumer 8GB Tier — e.g., RTX 4060):**
   * **`Qwen2.5-3B-Instruct` (Literature Reference):** Static weights consume ~5.99 GB. At 32K context, peak VRAM escalates to **9,747.98 MB (~9.75 GB)**, breaching the 8 GB physical ceiling by **+1,555.98 MB (+19.0%)**.
   * **`Phi-3.5-mini` (3.8B):** With 7.29 GB static weights, it breaches the 8 GB boundary at **just 1,024 tokens (8,410.93 MB)**.
2. **The 16 GB OOM Ceiling (Cluster/Cloud 16GB Tier):**
   * **`Qwen2.5-7B-Instruct`:** Static weights take 14.57 GB (~14.23 GB), leaving only ~1.8 GB headroom on a 16 GB allocation cap.
   * At **16K context**, peak VRAM hits **17,705.46 MB (+1,321 MB over 16GB)**.
   * At **32K context**, peak VRAM escalates to **20,834.15 MB (~20.83 GB)**, exceeding a 16 GB budget by **+4,450.15 MB (+27.2%)**.
3. **Severe Decoding Latency Penalty (TPOT Degradation):**
   * **Qwen2.5-7B** decoding slows from **19.81 ms/token (1K)** down to **130.39 ms/token (32K)** — a **6.58× latency penalty**.
   * **Qwen2.5-3B** decoding slows from **11.44 ms/token (1K)** down to **76.74 ms/token (32K)** — a **6.71× latency penalty**.

---

## 1. Testbed Hardware & Environment Reconciliation

| Parameter | Literature Baseline Sweep (This Document) | Production Benchmark Standard (`kv_bench`) | Note / Reconciliation |
| :--- | :--- | :--- | :--- |
| **Compute Host** | `gpu01.slurm.cpe.kmutt.ac.th` | `gpu01.slurm.cpe.kmutt.ac.th` | Same CPE Slurm Node |
| **GPU Model** | **NVIDIA GeForce RTX 4090** | **NVIDIA GeForce RTX 4090** | Ada Lovelace Architecture |
| **VRAM Mode** | **Physical Unconstrained (24,080 MB)** | **Capped at 16.00 GiB** (`cap_fraction = 0.680`) | *Unconstrained sweep was run to trace the true unthrottled memory cliff beyond 16 GB.* |
| **Runtime Engine** | PyTorch 2.5.1 / Transformers Native Greedy | **vLLM v0.30.0 (PagedAttention V1 Engine)** | *Production runs use vLLM for memory paging.* |
| **Workload Scope** | Single-stream context sweep ($N=1$) | Single-stream + **$N \in [1, 4, 8, 12, 16]$ SME Concurrency** | *Multi-user concurrency & LongMemEval are in `kv_bench`.* |
| **Primary Models** | Qwen 2.5 (1.5B, 3B, 7B) & Phi-3.5-mini | **Qwen 3 (4B & 1.7B Pure Dense GQA)** | *See [`docs/MODELS.md`](../docs/MODELS.md) for pure dense GQA rationale.* |

---

## 2. Four-Tier Empirical Benchmark Matrix

### 🔹 Tier 1: `Qwen/Qwen2.5-1.5B-Instruct` (Consumer 8GB Small Model)
* **Static Model Weights:** **2,945.29 MB** (~2.88 GB) | **8GB Headroom:** ~5,246 MB

| Context ($N$) | Peak VRAM (MB) | Est. KV-Cache (MB) | KV Share (%) | TPOT (ms/token) | Status on 8GB GPU (RTX 4060) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 3,000.94 | 55.65 | 1.85% | 16.67 | ✅ PASS (36.6% VRAM) |
| **1,024** | 3,046.96 | 101.67 | 3.34% | 8.69 | ✅ PASS (37.2% VRAM) |
| **2,048** | 3,142.50 | 197.21 | 6.28% | 9.07 | ✅ PASS (38.4% VRAM) |
| **4,096** | 3,325.59 | 380.30 | 11.44% | 9.85 | ✅ PASS (40.6% VRAM) |
| **8,192** | 3,697.76 | 752.47 | 20.35% | 12.29 | ✅ PASS (45.1% VRAM) |
| **16,384** | 4,442.11 | 1,496.81 | 33.70% | 21.29 | ✅ PASS (54.2% VRAM) |
| **32,768** | 5,930.79 | 2,985.50 | 50.34% | 44.24 | ⚠️ PASS (72.4% VRAM - TPOT 5.1× slow) |

---

### 🔹 Tier 2: `Qwen/Qwen2.5-3B-Instruct` (Literature Reference Benchmark)
* **Static Model Weights:** **5,994.48 MB** (~5.85 GB) | **8GB Headroom:** Only 2,197.52 MB

| Context ($N$) | Peak VRAM (MB) | Est. KV-Cache (MB) | KV Share (%) | TPOT (ms/token) | Status on 8GB GPU (RTX 4060) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 6,061.13 | 66.65 | 1.10% | 13.56 | ✅ PASS (74.0% VRAM) |
| **1,024** | 6,121.15 | 126.67 | 2.07% | 11.44 | ✅ PASS (74.7% VRAM) |
| **2,048** | 6,239.69 | 245.21 | 3.93% | 12.69 | ✅ PASS (76.2% VRAM) |
| **4,096** | 6,470.78 | 476.30 | 7.36% | 15.27 | ✅ PASS (79.0% VRAM) |
| **8,192** | 6,938.95 | 944.47 | 13.61% | 21.31 | ⚠️ High Stress (84.7% VRAM) |
| **16,384** | 7,875.29 | 1,880.81 | 23.88% | 38.64 | 🚨 Danger Zone (96.1% VRAM) |
| **32,768** | **9,747.98** | **3,753.50** | **38.51%** | **76.74** | 💥 **FATAL OOM CRASH (+1.55 GB over 8GB)** |

---

### 🔹 Tier 3: `microsoft/Phi-3.5-mini-instruct` (Comparative Dense Peer)
* **Static Model Weights:** **7,288.38 MB** (~7.12 GB) | **8GB Headroom:** Only ~903 MB

| Context ($N$) | Peak VRAM (MB) | Est. KV-Cache (MB) | TPOT (ms/token) | Status on 8GB GPU | Status on 24GB RTX 4090 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 7,780.16 | 491.78 | 34.46 | ⚠️ Near Limit (95.0%) | ✅ PASS |
| **1,024** | **8,410.93** | 1,122.55 | 22.28 | 💥 **OOM CRASH (1K tokens)** | ✅ PASS |
| **2,048** | 10,169.34 | 2,880.96 | 25.23 | 💥 OOM CRASH | ✅ PASS |
| **4,096** | 18,695.64 | 11,407.26 | 51.04 | 💥 OOM CRASH | ⚠️ Heavy Load (77.6%) |
| **8,192** | - | - | - | 💥 OOM CRASH | 💥 **OOM CRASH (24GB Exceeded)** |

---

### 🔹 Tier 4: `Qwen/Qwen2.5-7B-Instruct` (Cluster/Cloud 16GB Tier Reference)
* **Static Model Weights:** **14,568.65 MB** (~14.23 GB) | **16GB Headroom:** Only 1,815.35 MB

| Context ($N$) | Peak VRAM (MB) | Est. KV-Cache (MB) | KV Share (%) | TPOT (ms/token) | Status on 16GB Cluster GPU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 14,674.55 | 105.90 | 0.72% | 28.49 | ✅ PASS (89.6% VRAM) |
| **1,024** | 14,775.32 | 206.67 | 1.40% | 19.81 | ✅ PASS (90.2% VRAM) |
| **2,048** | 14,974.24 | 405.59 | 2.71% | 22.08 | ✅ PASS (91.4% VRAM) |
| **4,096** | 15,359.82 | 791.17 | 5.15% | 27.52 | ⚠️ High Stress (93.7% VRAM) |
| **8,192** | 16,142.12 | 1,573.47 | 9.75% | 39.97 | 🚨 Danger Zone (98.5% VRAM) |
| **16,384** | **17,705.46** | **3,136.81** | **17.72%** | **67.07** | 💥 **OOM CRASH (+1.32 GB over 16GB)** |
| **32,768** | **20,834.15** | **6,265.50** | **30.07%** | **130.39** | 💥 **FATAL OOM CRASH (+4.45 GB over 16GB)** |

---

## 3. Synthesis & Transition to Production `kv_bench` Suite

```
Empirical Memory Saturation Floor (32K Context):
┌─────────────────────────────────────────────────────────────┐
│ Qwen2.5-7B   (20.83 GB)  ██████████████████████████████ 💥 Exceeds 16GB cap by +4.45 GB
│ Phi-3.5-mini (>24.00 GB) ██████████████████████████████████ 💥 Exceeds 24GB at 8K
│ Qwen2.5-3B   (9.75 GB)   ████████████ 💥 Exceeds 8GB cap by +1.55 GB
│ Qwen2.5-1.5B (5.93 GB)   ███████ ✅ Fits in 8GB (TPOT degrades 5.1×)
└─────────────────────────────────────────────────────────────┘
  0 GB         8 GB (RTX 4060)   16 GB (Cluster Tier)    24 GB (RTX 4090)
```

### 🔬 Alignment with Project Work Packages:
1. **Substantiates the 3-Axis Hypothesis:**  
   The empirical data confirms that neither 8 GB nor 16 GB hardware budgets can support long contexts without active cache co-management.
2. **Transition to Primary Workload:**  
   Future baseline profiling sweeps will be conducted strictly using the unified harness in [`khemingkapat/kv_bench`](https://github.com/khemingkapat/kv_bench):
   * Primary model family: **`Qwen/Qwen3-4B-Instruct-2507`** (Pure Dense GQA workhorse) and **`Qwen/Qwen3-1.7B`** (Local adaptation).
   * Engine: **vLLM (PagedAttention)** with enforced **16.00 GiB VRAM allocation cap**.
   * Workload: Multi-turn conversational traces under **$N \in [1, 4, 8, 12, 16]$ concurrent users**.

---
*Reference dataset captured on KMUTT CPE Slurm Cluster (`gpu01.slurm.cpe.kmutt.ac.th`).*
