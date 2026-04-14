# Progress Ratio Embeddings (PRE)

> **Findings of ACL 2026** · [arXiv:2512.06938](https://arxiv.org/abs/2512.06938)

Official implementation of **Progress Ratio Embeddings**, a continuous positional signal for precise, generalisable length control in autoregressive text generation.

**Authors:** Ivanhoé Botcazou · Tassadit Amghar · Sylvain Lamprier · Frédéric Saubion  
**Affiliation:** LERIA, University of Angers, France

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Inference with PRE-BART (HuggingFace)](#inference-with-pre-bart-huggingface)
  - [Training from Scratch](#training-from-scratch)
  - [Evaluation](#evaluation)
- [Supported Models & Baselines](#supported-models--baselines)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

---

## Overview

None Instruct Language Models struggle to respect user-specified output lengths, especially for lengths never seen during training. Existing approaches such as Reverse Positional Embeddings (RPE) encode discrete countdown signals to control the generation over an exoected target length. However that control degrades sharply on out of distribution target length.

**PRE** replaces this discret countdown with a continuous *progress ratio* $r_t = t / l \in [0, 1]$, where $t$ is the current decoding step and $l$ is the target length. This ratio is encoded as a sinusoidal embedding whose frequency grows with $r_t$, creating an *impatience signal* that tells the model how far along generation should be:

$$
\xi(r)_j =
\begin{cases}
\sin(\omega_r \cdot x_j) & \text{if } j \text{ is even} \\
\cos(\omega_r \cdot x_j) & \text{if } j \text{ is odd}
\end{cases},
\quad
\omega_r = r \cdot \frac{d_{\text{model}}}{2}
$$

The maximum frequency is bounded by the Nyquist criterion, ensuring the signal is perfectly representable without aliasing for any target length.

---

## Key Results

### Summarization — CNN/DailyMail

| Model | R-1 | R-2 | R-L | BERTScore | MAE ± SD |
|---|---|---|---|---|---|
| BART-Large | 44.2 | 21.1 | 40.9 | 69.7 | 19.2 ± 17.0 |
| RPE-BART-Large | 44.5 | 21.2 | 41.3 | 69.4 | 1.6 ± 3.6 |
| **PRE-BART-Large** | **45.3** | **21.9** | **42.2** | **69.8** | **0.5 ± 0.3** |

### Out-of-Distribution Generalization (CNN/DailyMail, lengths unseen at training)

| Target length range | % Outliers — RPE | % Outliers — PRE |
|---|---|---|
| [300, 350) | 10.8% | **0.4%** |
| [400, 450) | 22.3% | **1.5%** |
| [500, 550) | 37.0% | **4.3%** |
| [650, 700) | 53.9% | **7.4%** |
| [800, 850) | 75.1% | **9.0%** |
| [950, 1000) | 95.8% | **19.8%** |

### Question Generation — SQuAD

| Model | BLEU | R-1 | R-L | BERTScore | MAE ± SD |
|---|---|---|---|---|---|
| BART-Large | 16.7 | 54.4 | 50.1 | **75.2** | 3.12 ± 3.3 |
| RPE-BART-Large | 16.4 | 52.1 | 47.7 | 72.6 | 0.80 ± 3.6 |
| **PRE-BART-Large** | **18.6** | **55.3** | **50.8** | 74.8 | **0.0 ± 0.01** |

---

## Repository Structure

```
src/
├── BART/                        # Encoder-decoder models (BART-based)
│   ├── PRE_BART/                # Progress Ratio Embeddings (proposed)
│   ├── RPE_BART/                # Reverse Positional Embeddings (baseline)
│   └── LRPE_BART/               # Length Ratio Positional Embeddings (baseline)
├── T5/
│   └── PRE_T5/                  # PRE applied to T5
├── LLAMA/
│   └── PRE_LLAMA/               # PRE applied to LLaMA (decoder-only)
├── DATASETS/
│   ├── CNN/                     # CNN/DailyMail dataset loader
│   ├── XSUM/                    # XSum dataset loader
│   └── SQUAD/                   # SQuAD question generation loader
└── UTILS/
    ├── modeling_trainer_encoder_decoder.py   # Training loop for seq2seq models
    ├── modeling_trainer_decoder_only.py      # Training loop for decoder-only models
    ├── generate_and_evaluation_len_enc_dec.py
    ├── finesure_evaluation.py
    ├── length_statistic.py
    ├── model_description.py
    └── download_model.py
```

---

## Installation

**Python 3.10+ recommended.**

```bash
git clone git@github.com:Ivanbtz9/ProgressRatioEmbeddings.git
cd ProgressRatioEmbeddings
pip install -r requirements.txt
```

For LLaMA experiments, use the dedicated environment:

```bash
cd src/LLAMA
conda env create -f env.yml
conda activate pre_llama_env
```

Core dependencies: `transformers`, `torch`, `datasets`, `evaluate`, `rouge_score`, `bert_score`.

---

## Quick Start

### Inference with PRE-BART (HuggingFace Hub)

A fine-tuned PRE-BART-Large checkpoint on CNN/DailyMail is available on the Hugging Face Hub:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
    "Ivanhoe9/prebart-large-cnn",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "Ivanhoe9/prebart-large-cnn",
    trust_remote_code=True,
)

article = "Your input article here..."
inputs = tokenizer([article], return_tensors="pt", truncation=True, max_length=1024)

# Specify your desired output length (in tokens), e.g. 80
target_len = torch.tensor([80])

summary_ids = model.generate(
    inputs["input_ids"],
    target_len=target_len,
    **model.config.task_specific_params["summarization"],
)

summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
print(summary)
```

The `target_len` argument is the **only** change relative to standard BART inference.

## Supported Models & Baselines

| Model class | Architecture | Description |
|---|---|---|
| `PRE_BART` | Encoder-decoder | **Proposed method.** PRE on BART-Large. |
| `RPE_BART` | Encoder-decoder | Baseline: Reverse Positional Embeddings. |
| `LRPE_BART` | Encoder-decoder | Baseline: Length Ratio Positional Embeddings. |
| `PRE_T5` | Encoder-decoder | PRE applied to T5. |
| `PRE_LLAMA` | Decoder-only | PRE applied to LLaMA. Prompt tokens receive $\xi(0)$; target tokens receive increasing ratios. |

All models share the same `target_len` interface at inference time.

---

## Datasets

Dataset loaders in `src/DATASETS/` wrap HuggingFace `datasets` and return the fields expected by the training utilities.

| Module | Task | Reference |
|---|---|---|
| `DATASETS/CNN` | Abstractive summarization | CNN/DailyMail (Nallapati et al., 2016) |
| `DATASETS/XSUM` | Highly abstractive summarization | XSum (Narayan et al., 2018) |
| `DATASETS/SQUAD` | Question generation | SQuAD (Rajpurkar et al., 2016) |

---

## Citation

If you use this code or the PRE method in your research, please cite:

```bibtex
@inproceedings{botcazou2026pre,
  title     = {Progress Ratio Embeddings: An Impatience Signal for Robust Length Control in Neural Text Generation},
  author    = {Botcazou, Ivanho{\'e} and Amghar, Tassadit and Lamprier, Sylvain and Saubion, Fr{\'e}d{\'e}ric},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.06938},
}
```

---

## License

This project is licensed under the Apache License. See [LICENSE](LICENSE) for details.
