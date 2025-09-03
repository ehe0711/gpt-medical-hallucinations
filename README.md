# Benchmarking Hallucination Detection and Calibration in Medical LLMs

This repository contains the code, prompts, and manuscript for **Benchmarking Hallucination Detection and Calibration in Medical LLMs**, a project evaluating hallucination behavior in **GPT-4o-mini** and **GPT-5-mini** on the MedQuAD biomedical QA dataset.

---

## 📌 Project Overview
Large language models (LLMs) show strong potential in medicine but remain prone to **hallucinations**.  
This study evaluates:
- **Domain-level performance** — which biomedical sectors are most error-prone.  
- **Cross-model evaluation** — where GPT-5 corrects GPT-4’s errors (and where it regresses).  
- **Confidence calibration** — whether self-reported confidence tracks factuality.  
- **Confidence alignment** — correlation between self-confidence and external NLI entailment.  

We benchmark hallucination rates, analyze confidence reliability, and provide reproducible scripts for replication.

---

## 📂 Repository Structure
```
├─ code/          # Core analysis scripts
├─ data/          # Dataset instructions + small samples (no raw MedQuAD)
├─ materials/     # All outputs: figures, tables, and result files
└─ paper/         # Manuscript sources (Word, LaTeX, references)
```

**Key scripts:**
- `dataset_cleaning.py` – Prepares raw MedQuAD for model input.  
- `dataset_creation.py` – Creates standardized dataset splits.  
- `gpt4_responses.py` / `gpt5_responses.py` – Collect model responses.  
- `gpt4_hallucination_coding.py` / `gpt5_hallucination_coding.py` – Apply hallucination coding.  
- `hallucination_analysis.py` – Aggregate metrics & generate figures.  

---

## 🔧 Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/med-llm-hallucination-benchmark.git
cd med-llm-hallucination-benchmark
conda env create -f environment.yml
conda activate med-llm
```

Set your API key (if running inference with OpenAI models):

```bash
export OPENAI_API_KEY=your_key_here
```

---

## 🚀 Usage
### Run full pipeline:
```bash
# Generate responses
python code/gpt4_responses.py
python code/gpt5_responses.py

# Apply hallucination coding
python code/gpt4_hallucination_coding.py
python code/gpt5_hallucination_coding.py

# Analyze & produce figures
python code/hallucination_analysis.py
```

Outputs (tables and figures) will be stored in `materials/`.

---

## 📊 Results
- GPT-5-mini reduced hallucinations from **49.6% → 31.1%**.  
- GPT-5 corrected **28.5% of GPT-4 errors**, but regressed in ~10%.  
- Calibration improved, with fewer high-confidence hallucinations.  
- Overconfidence persisted — confidence alone is insufficient as a safety metric.  
- Dataset created of prompts where both GPT-4 and GPT-5 hallucinated; can be accessed at https://huggingface.co/datasets/ehe07/gpt-failure-cases-dataset

All final figures and tables are available in the `materials/` folder.

---

## 📑 Citation
If you use this work, please cite:

```bibtex
@article{he2025hallucinations,
  title={Benchmarking Hallucination Detection and Calibration in Medical LLMs},
  author={He, Ethan},
  year={2025},
  note={Preprint}
}
```

---

## 📜 License
- **Code**: MIT License  
- **Text/Manuscript**: CC BY 4.0  
