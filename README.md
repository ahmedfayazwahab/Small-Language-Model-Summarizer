# Training Small Models That Actually Work

**A repeatable SFT → DPO pipeline for a Qwen3-0.6B documentation summarizer — and what it revealed about measuring model quality.**

AI Capstone Project · Team **Outliers** · Humber Polytechnic (FAST ICT) · Feb–Aug 2026

---

## What this is

This repo holds the configs, datasets, and final report for a capstone project that answered one question:

> Can a *very small* language model (~0.6B parameters), trained on a single shared GPU through a **repeatable process**, do a real job reliably — instead of paying per call for a large commercial LLM?

The job we picked was **document summarization**: given a page of technical documentation, produce a faithful summary of 150 words or less. It was chosen because it is concrete, cheap to evaluate, and small enough to iterate on constrained hardware. The real deliverable isn't the summarizer — it's a pipeline that can turn out small, specialized models cheaply, with predictable quality.

We built an end-to-end **Supervised Fine-Tuning (SFT) → Direct Preference Optimization (DPO)** pipeline on top of an 8-bit **Qwen3-0.6B** base using **LoRA** adapters, ran **47 documented training runs**, and shipped a tuned production configuration plus export-ready adapters for local/edge serving via Ollama.

> **Note:** This project was carried out for an industry client under a non-disclosure agreement. The client and their internal training platform are **not named** anywhere in this repo, by design. Technical findings are the team's own observations.

---

## Repository structure

```
capstone-slm-summarizer/
├── README.md
├── report/
│   ├── AI_Capstone_Final_Report.pdf     ← start here (readable in-browser)
│   └── AI_Capstone_Final_Report.docx    ← same report, editable source
├── configs/
│   └── dpo/
│       └── experiments/                 ← the single-lever DPO sweep (one knob changed per run)
│           ├── run1_baseline_beta0.15.yaml
│           ├── run2_beta0.10.yaml
│           ├── run3_lora_r16.yaml
│           ├── run4_2epochs.yaml        ← the winner
│           ├── run5_ipo.yaml
│           └── run6_beta0.10_r16.yaml
└── data/
    ├── sft/
    │   ├── sft_train_v3.jsonl           ← 100 rows · supervised training
    │   └── sft_eval_v3.jsonl            ←  25 rows · supervised held-out eval
    └── dpo/
        ├── dpo_train.jsonl              ← 200 rows · preference training pairs
        └── dpo_heldout_eval.jsonl       ←  60 rows · frozen preference held-out set
```

---

## The data

All records derive from **publicly accessible vendor API documentation** (crawled and cleaned), with **synthetic reference summaries**. No client data and no personal data were used. Datasets are provided for research/education; consult each source vendor's terms before any redistribution or commercial use.

**SFT files** — `{ prompt, completion }`
The model learns the task from a document `prompt` and a good `completion` summary.
Sources: OpenAI, Llama, and DeepSeek docs. Train/eval URL sets are verified non-overlapping.

**DPO files** — `{ prompt, chosen, rejected }`
The model learns *preferences* — the `chosen` summary is good; the `rejected` one is a **deliberately degraded** version carrying exactly one controlled defect (factual error, wrong scope, vague/generic, key omission, verbose padding, or leaked reasoning).

| File | Role | Rows | Notes |
|---|---|---|---|
| `data/sft/sft_train_v3.jsonl` | SFT — train | 100 | OpenAI / Llama / DeepSeek docs |
| `data/sft/sft_eval_v3.jsonl` | SFT — **held-out eval** | 25 | disjoint from train |
| `data/dpo/dpo_train.jsonl` | DPO — train | 200 | preference pairs |
| `data/dpo/dpo_heldout_eval.jsonl` | DPO — **held-out eval** | 60 | 43 unique docs; 26 deliberately out-of-domain to test transfer |

---

## The DPO experiment sweep (`configs/dpo/experiments/`)

The project's defining method: **change exactly one lever per run** from a fixed baseline, so every effect is attributable to a known cause. Across six runs spanning four levers, only two levers moved anything.

| Config | Lever changed vs. baseline | Verdict |
|---|---|---|
| `run1_baseline_beta0.15.yaml` | — (anchor: beta 0.15, r8, 1 epoch) | Underfit — policy never moved |
| `run2_beta0.10.yaml` | beta 0.15 → **0.10** | First run to learn anything |
| `run3_lora_r16.yaml` | LoRA r8 → **r16** (at beta 0.15) | Underfit — added capacity didn't help |
| `run4_2epochs.yaml` | **2 epochs** (at beta 0.10) | **✅ Winner** — margin multiplied ~6× |
| `run5_ipo.yaml` | sigmoid → **IPO** loss (at beta 0.10) | No traction |
| `run6_beta0.10_r16.yaml` | LoRA r8 → **r16** (at beta 0.10) | Underfit — added capacity didn't help |

**Takeaway:** `beta 0.15` is a dead zone; `beta 0.10` is the threshold where the policy can move at all; a **second epoch** was the single most effective change in the whole programme. Adapter rank and IPO loss did nothing.

Each config targets a base at `/models/base-models/qwen3-0-6b` and continues from SFT adapters; the `/workspace/<...>` paths are platform placeholders. No secrets are stored — credentials are supplied via the `HF_TOKEN` environment variable.

---

## Key results

- **SFT:** solved and reliable — **100% pass** on the largest held-out set (65/65) and 24–25 / 25 on the standard set.
- **DPO:** tuned to a defensible production config; the strongest run reached **93% held-out reward accuracy** with no train-vs-test gap.
- **The pivotal finding:** the platform's default evaluation metric (embedding similarity) **could not tell a good model from a bad one** — it ranked a clearly underfit model (81.7%) *above* the tuned winner (80.0%). The fix — **held-out reward accuracy**, computable from data already being logged — turned a flat line into a 30-point spread.
- **The real bottleneck** is data scale and evaluation quality, **not** hyperparameter tuning. On small data, *the dataset is the hyperparameter*.

Full details, tables, and the four-phase automation roadmap are in **`report/AI_Capstone_Final_Report.pdf`**.

---

## Team Outliers

Sean Cheung (team lead) · Ahmed Fayaz Wahab · Yash Rana · Khirod Kumar Ardi

With thanks to our faculty mentors and to Humber Polytechnic's FAST ICT Capstone program.
# capstone-slm-summarizer
# capstone-slm-summarizer
