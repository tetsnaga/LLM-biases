<!-- # LLM-biases
Cognitive Bias Quantification and Propagation in LLMs

Quick Review:<br>
    1. Run the requirements.txt file to install required depenedencies (pip install -r requirements.txt)<br>
    2. The Persona file (sorted_personas.csv) and dataset (climate-fever-dataset.json) should be under Data/ folder.<br>
    3. All the results will go to outputs/ folder.<br>
    4. Template of the prompts are available in the prompts.txt file. 

To run the code for a specific persona file and claim do the following:

    1. Geenerate the claims for each evidence level using sample_claims.py OR just use example_usage.py to generate all of them which will be under Data directory.
    2. Run `chmod +x experiment.py`<br> in project directory.
    3. Run `python experiment.py --model [model name] --personas [path to the personas] --claims [path to claims] --evidenceLevel [evidencelevel 1,2,3 or 4] --n_evidence [between 1 to 5]`<br>

For example command for evidece level claims will be:<br>

`python experiment.py --model deepseek-r1:1.5b  --claims Data/claims_EL1.json  --evidenceLevel 1  --n_evidence 3 --personas Data/personas.csv`

# Command Line Arguments

| Argument | Requirement | Default Value | Description |
| :--- | :---: | :--- | :--- |
| **`--model`** | Required | N/A | Name of the **Ollama model** to use. |
| **`--evidenceLevel`** | Required | N/A | Determines which **prompt template** to use: 1 or 2 or 3 or 4. |
| **`--personas`** | Required | `Data/synthetic_climate_personas.csv` | Path to the **personas dataset** (CSV) containing demographic, cognitive, and belief attributes. |
| **`--claims`** | Required | `Data/climate-fever-dataset.csv` | Path to the **claims dataset** (JSON) containing the claim text and evidence entries (claims_EL1.json, claims_EL2.json, claims_EL3.json, claims_EL4.json) |
| **`--n_evidence`** | Optional | `1` | **Number of evidence snippets** to include (for evidence levels 2–4). |
| **`--outdir`** | Optional | `outputs` | **Output folder** where model results (CSV) will be saved. |
| **`--temperature`** | Optional | `0.2` | Controls **sampling randomness** (Lower = more deterministic). |
| **`--n_personas`** | Optional | All | **Number of personas** to sample (If not provided, all are used). |
| **`--n_claims`** | Optional | All | **Number of claims** to sample (If not provided, all are used). |
| **`--evidence_random`** | Required | `False` | If `True`, randomly samples `n_evidence` from each claim’s list instead of using the first ones. | -->


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

This creates `personas.csv` in the `Data/` directory.

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
  --personas Data/personas.csv
```

---

## ⚙️ Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | ✅ Yes | — | Ollama model name to use |
| `--evidenceLevel` | ✅ Yes | — | Evidence level (1, 2, 3, or 4) determines prompt template |
| `--personas` | ✅ Yes | `Data/sorted_personas.csv` | Path to personas CSV file - use Data/personas.csv |
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
