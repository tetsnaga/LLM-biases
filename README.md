# 🧠 LLM-biases

> **Cognitive Bias Quantification and Propagation in Large Language Models**

A research tool for analyzing how cognitive biases manifest and propagate through LLM responses when presented with climate-related claims across different evidence levels.

---

## 🚀 Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Project Structure

Ensure your project follows this structure:

```
LLM-biases/
├── Data/
│   ├── climate-fever-dataset.json
│   └── claims_EL*.json (generated)
├── outputs
└── prompts.txt
```

---

## 📋 Setup Instructions

### 1️⃣ Generate Claims and Personas

For claims You have two options:

**Option A:** Generate claims for a specific evidence level
```bash
python sample_claims.py
```

**Option B:** Generate all claims at once (recommended)
```bash
python example_usage.py
```

This creates `claims_EL1.json`, `claims_EL2.json`, `claims_EL3.json`, and `claims_EL4.json` in the `Data/` directory.


For creating Personas:
```bash
python sample_personas.py
```

This creats `sample_personas.csv` in the `Data/` directory.

### 2️⃣ Set Permissions

```bash
chmod +x experiment.py
```

### 3️⃣ Run Experiment

```bash
python experiment.py \
  --model <model_name> \
  --personas <path_to_personas> \
  --claims <path_to_claims> \
  --evidenceLevel <1|2|3|4> \
  --n_evidence <1-5>
```

---

## 💡 Example Usage

Run an experiment with Evidence Level 1:

```bash
python experiment.py \
  --model deepseek-r1:1.5b \
  --claims Data/claims_EL1.json \
  --evidenceLevel 1 \
  --n_evidence 3 \
  --personas Data/sampled_personas.csv
```

---

## ⚙️ Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | ✅ Yes | — | Ollama model name to use |
| `--evidenceLevel` | ✅ Yes | — | Evidence level (1, 2, 3, or 4) determines prompt template |
| `--personas` | ✅ Yes | `Data/synthetic_climate_personas.csv` | Path to personas CSV file - use Data/personas.csv |
| `--claims` | ✅ Yes | `Data/climate-fever-dataset.csv` | Path to claims JSON file |
| `--n_evidence` | ❌ No | `1` | Number of evidence snippets (for levels 2–4) |
| `--outdir` | ❌ No | `outputs` | Output directory for results |
| `--temperature` | ❌ No | `0.2` | Sampling temperature (lower = more deterministic) |
| `--n_personas` | ❌ No | All | Number of personas to sample |
| `--n_claims` | ❌ No | All | Number of claims to sample |
| `--evidence_random` | ❌ No | `False` | Randomly sample evidence instead of using first ones |

---

## 📂 Output

All experimental results are saved to the `outputs/` folder as CSV files containing:
- Model responses
---
