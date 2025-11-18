# counterfactual_explanations
Benchmarking counterfactual explanation algorithms and evaluating them using human-aligned LLMs (CounterEval).

## Setup

All commands should be run from the **project root directory**.

This project requires **two separate Python environments**:

1. A **Python 3.7.9 environment** for generating counterfactuals using the CARLA framework  
2. A **modern Python environment** (e.g., Python 3.13) for evaluation and analysis

Both environments share the same dataset and model files.

---

### 1. Fetch datasets and pretrained models

Before creating environments, run:

```bash
python src/utils/fetch_data_and_models.py
```

This downloads the CounterEval dataset and their Llama 3.1 8B model weights.

---

### 2. Counterfactual generation environment (Python 3.7.9)

Create and activate a Python 3.7.9 environment:

Install required packages **in the following order**:

1. Compatibility packages:

```bash
python -m pip install "typing-extensions==4.7.1" "numpy==1.19.4"
```

2. Correct PyTorch version:

```bash
python -m pip install torch===1.7.1 torchvision===0.8.2
```

3. CARLA and all remaining dependencies:

```bash
pip install -r requirements_carla.txt
```

This environment is used **only to generate counterfactuals**.

Example usage:

```bash
python -m src.generate.main_generate --algorithm FACE --n_factuals 50
```

Counterfactuals are saved to:

```
data/cf_raw/
```

---

### 3. Evaluation environment (modern Python)

Create and activate a modern Python (e.g. 3.13) environment:

Install evaluation dependencies:

```bash
pip install -r requirements_eval.txt
```

This environment is used for:

- CounterEval evaluation
- LLM scoring
- Analysis and results
