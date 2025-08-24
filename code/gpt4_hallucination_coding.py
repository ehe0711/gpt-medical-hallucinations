"""
Analyze GPT responses for hallucinations using NLI
"""

print("Starting script...")
try:
    print("Importing required packages...")
    import pandas as pd
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from tqdm import tqdm
    from typing import List, Dict, Tuple
    import os
    print("All packages imported successfully")
except Exception as e:
    print(f"Error importing packages: {str(e)}")
    raise
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import os

# Medical synonyms dictionary for improved matching
MEDICAL_SYNONYMS = {
    # Common symptom synonyms
    "pain": ["discomfort", "ache", "soreness", "tenderness", "distress", "agony", "suffering"],
    "fatigue": ["exhaustion", "tiredness", "lethargy", "weakness", "malaise", "weariness", "burnout"],
    "inflammation": ["swelling", "edema", "redness", "irritation", "inflammatory response"],
    "fever": ["pyrexia", "hyperthermia", "elevated temperature", "febrile response"],
    "nausea": ["queasiness", "sickness", "vomiting", "emesis", "gastrointestinal upset"],
    
    # Disease synonyms
    "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma", "malignant growth", "oncological condition"],
    "heart attack": ["myocardial infarction", "cardiac arrest", "coronary thrombosis", "MI", "acute coronary syndrome"],
    "stroke": ["cerebrovascular accident", "CVA", "brain attack", "cerebral infarction", "cerebral hemorrhage"],
    "diabetes": ["diabetes mellitus", "DM", "hyperglycemia", "blood sugar disorder"],
    "hypertension": ["high blood pressure", "HTN", "elevated blood pressure", "arterial hypertension"],
    
    # Anatomical synonyms
    "heart": ["cardiac", "myocardium", "cardiovascular", "coronary", "cardiopulmonary"],
    "brain": ["cerebral", "neural", "cerebrum", "neurological", "intracranial"],
    "stomach": ["gastric", "abdominal", "belly", "gastrointestinal", "digestive"],
    "lungs": ["pulmonary", "respiratory", "bronchial", "thoracic", "airways"],
    "liver": ["hepatic", "hepatobiliary", "biliary system"],
    
    # Treatment synonyms
    "surgery": ["operation", "procedure", "surgical intervention", "surgical procedure", "operative treatment"],
    "medication": ["medicine", "drug", "pharmaceutical", "treatment", "therapeutic agent", "pharmacotherapy"],
    "therapy": ["treatment", "intervention", "therapeutic", "rehabilitation", "therapeutic intervention"],
    "chemotherapy": ["chemo", "cytotoxic therapy", "anticancer treatment", "antineoplastic therapy"],
    "radiation": ["radiotherapy", "radiation therapy", "therapeutic radiation", "RT"],
    
    # Diagnostic synonyms
    "examination": ["assessment", "evaluation", "workup", "exam", "clinical examination", "medical evaluation"],
    "test": ["investigation", "analysis", "screening", "diagnostic", "laboratory test", "clinical test"],
    "scan": ["imaging", "radiograph", "diagnostic imaging", "medical imaging", "radiological examination"],
    "biopsy": ["tissue sample", "pathological examination", "histological examination"],
    "monitoring": ["observation", "surveillance", "tracking", "follow-up"],
    
    # Medical specialties
    "cardiology": ["heart specialist", "cardiovascular medicine", "heart medicine"],
    "oncology": ["cancer treatment", "cancer medicine", "cancer care"],
    "neurology": ["brain medicine", "neurological medicine", "nerve specialist"],
    
    # Medical conditions
    "infection": ["infectious disease", "pathogenic invasion", "microbial infection"],
    "allergy": ["allergic reaction", "hypersensitivity", "allergic response"],
    "autoimmune": ["immune disorder", "autoimmune disease", "immune-mediated condition"]
}

# Load NLI model and tokenizer
print("Loading NLI model...")
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Map label indices to their names
LABEL_NAMES = ["contradiction", "entailment", "neutral"]

# Batch sizes for efficient processing
BATCH_SIZE = 10  # Process 10 rows at a time

def expand_with_synonyms(text: str) -> str:
    """
    Expand a text with medical synonyms to improve matching
    """
    words = text.lower().split()
    expanded_words = words.copy()
    
    for word in words:
        if word in MEDICAL_SYNONYMS:
            expanded_words.extend(MEDICAL_SYNONYMS[word])
    
    return " ".join(expanded_words)

def get_batch_nli_predictions(expert_answers: List[str], llm_responses: List[str]) -> List[Dict]:
    """
    Get NLI predictions for a batch of text pairs
    Returns list of dicts with labels, scores, and hallucination flags
    """
    # Expand all texts with synonyms
    expert_expanded = [expand_with_synonyms(ans) for ans in expert_answers]
    llm_expanded = [expand_with_synonyms(resp) for resp in llm_responses]
    
    # Encode the batch
    encoding = tokenizer(
        expert_expanded,
        llm_expanded,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move tensors to device
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # Get predictions for whole batch
    with torch.no_grad():
        outputs = model(**encoding)
        batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        results = []
        for probs in batch_probs:
            label_idx = torch.argmax(probs).item()
            label = LABEL_NAMES[label_idx]
            
            results.append({
                "label": label,
                "scores": {name: score.item() for name, score in zip(LABEL_NAMES, probs)},
                "is_hallucination": label in ["contradiction", "neutral"]
            })
        
        return results

def analyze_hallucinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze LLM responses for hallucinations using NLI model
    Returns DataFrame with added columns for hallucination classification
    """
    # Add required columns if they don't exist
    if 'is_hallucination' not in df.columns:
        df['is_hallucination'] = 0
    if 'classification' not in df.columns:
        df['classification'] = ''
    if 'confidence_scores' not in df.columns:
        df['confidence_scores'] = None
        
    # Process in batches
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_start in tqdm(range(0, len(df), BATCH_SIZE), total=total_batches, desc="Analyzing responses"):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch = df.iloc[batch_start:batch_end]
        
        # Prepare batch data
        expert_answers = batch['answer'].astype(str).tolist()
        llm_responses = batch['gpt4_response'].astype(str).tolist()
        results = get_batch_nli_predictions(expert_answers, llm_responses)
        
        # Update DataFrame with results
        for i, (idx, result) in enumerate(zip(batch.index, results)):
            df.loc[idx, 'is_hallucination'] = int(result['is_hallucination'])
            df.loc[idx, 'classification'] = result['label']
            df.loc[idx, 'confidence_scores'] = str(result['scores'])
    
    return df

def main():
    try:
        # Read the dataset
        print("Reading dataset from ../data/gpt4_responses.csv...")
        print("Current working directory:", os.getcwd())
        df = pd.read_csv('../data/gpt4_responses.csv')
        print(f"Loaded dataset with {len(df)} rows")
    except Exception as e:
        print(f"Error reading dataset: {str(e)}")
    
    # Analyze hallucinations
    print("Analyzing hallucinations...")
    df = analyze_hallucinations(df)
    
    # Save results with hallucination annotations
    print("\nSaving results...")
    df.to_csv('../data/gpt4_responses_with_hallucinations.csv', index=False)
    print("Saved results with hallucination annotations (1=hallucination, 0=not hallucination)")
    
    # Print summary statistics
    print("\nAnalysis Summary:")
    print(f"Total responses analyzed: {len(df)}")
    print(f"Total hallucinations found: {df['is_hallucination'].sum()}")
    print(f"Hallucination rate: {(df['is_hallucination'].sum() / len(df)) * 100:.2f}%")
    
    # Print breakdown by classification
    print("\nBreakdown by classification:")
    label_counts = df['classification'].value_counts()
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(df)*100:.2f}%)")

if __name__ == "__main__":
    main()
