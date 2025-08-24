"""
Analyze GPT responses for hallucinations using NLI
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple

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

# Constants for processing
BATCH_SIZE = 64  # Larger batch size for better GPU utilization
MAX_LENGTH = 512  # Maximum sequence length for the model
NUM_WORKERS = 4  # Number of workers for data loading

# Force CUDA setup and optimization
assert torch.cuda.is_available(), "CUDA must be available!"
torch.cuda.empty_cache()
DEVICE = torch.device("cuda:0")  # Explicitly use first GPU
BATCH_SIZE = 32  # Balanced batch size for GPU memory

# Enable CUDA optimizations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")

# Configure CUDA settings
torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
torch.cuda.empty_cache()

# Verify CUDA is available
assert torch.cuda.is_available(), "CUDA must be available"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

# Load and optimize model for GPU
print("Loading NLI model...")
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Move model to GPU and optimize
print("Moving model to GPU and optimizing...")
model = model.cuda()  # Explicitly move to CUDA
model = model.half()  # Convert to FP16 for faster inference
model.eval()  # Set to evaluation mode

# Verify model is on CUDA
assert next(model.parameters()).is_cuda, "Model must be on CUDA!"

LABEL_NAMES = ["contradiction", "entailment", "neutral"]

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
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    
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

@torch.no_grad()
def process_batch(expert_answers: list, llm_responses: list) -> tuple:
    """Process a batch of responses using GPU acceleration"""
    # This function is kept for backward compatibility but not used in the updated analyze_hallucinations
    # Tokenize and move to GPU
    inputs = tokenizer(
        expert_answers,
        llm_responses,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    # Explicitly move all tensors to GPU
    cuda_inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Run model with mixed precision
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(**cuda_inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_indices = torch.argmax(probs, dim=1)
        confidences = probs.gather(1, pred_indices.unsqueeze(1)).squeeze()
    
    # Move results back to CPU for numpy conversion
    pred_indices = pred_indices.cpu()
    confidences = confidences.cpu()
    
    # Convert to numpy arrays
    labels = [LABEL_NAMES[i] for i in pred_indices.numpy()]
    scores = confidences.numpy()
    
    # Clear GPU cache if needed
    if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
        torch.cuda.empty_cache()
    
    return labels, scores

from contextlib import nullcontext
import math

@torch.no_grad()
def analyze_hallucinations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze responses for hallucinations using GPU-accelerated inference"""
    total_samples = len(df)
    print(f"Starting analysis of {total_samples} responses with batch size {BATCH_SIZE}")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    # We'll store the results in these lists
    classifications = []
    confidence_dicts = []  # To store all three confidence scores
    
    try:
        for i in tqdm(range(0, total_samples, BATCH_SIZE)):
            # Process current batch
            batch = df.iloc[i:i + BATCH_SIZE]
            expert_answers = batch['answer'].astype(str).tolist()
            llm_responses = batch['gpt5_response'].astype(str).tolist()
            
            # Process with full probabilities for all classes
            # Encode the batch
            encoding = tokenizer(
                expert_answers,
                llm_responses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move tensors to device
            encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
            
            # Get predictions for whole batch
            with torch.cuda.amp.autocast():
                outputs = model(**encoding)
                batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                batch_results = []
                batch_class = []
                
                for probs in batch_probs:
                    label_idx = torch.argmax(probs).item()
                    label = LABEL_NAMES[label_idx]
                    batch_class.append(label)
                    
                    # Store all three probability scores in dictionary format
                    scores_dict = {name: score.item() for name, score in zip(LABEL_NAMES, probs)}
                    batch_results.append(scores_dict)
            
            # Store results
            classifications.extend(batch_class)
            confidence_dicts.extend(batch_results)
            
            # Monitor GPU memory
            if i > 0 and i % (BATCH_SIZE * 5) == 0:
                torch.cuda.empty_cache()
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory. Try reducing batch size.")
        raise e
    
    # Update DataFrame with results
    df['classification'] = classifications
    # Store full confidence scores dictionary as string
    df['confidence_scores'] = [str(d) for d in confidence_dicts]
    # Add individual confidence scores
    df['entailment_prob'] = [d.get('entailment', 0) for d in confidence_dicts]
    df['contradiction_prob'] = [d.get('contradiction', 0) for d in confidence_dicts]
    df['neutral_prob'] = [d.get('neutral', 0) for d in confidence_dicts]
    # Store primary confidence score (matching GPT-4 format) as the score of the predicted class
    df['gpt5_confidence'] = [d.get(c, 0) for d, c in zip(confidence_dicts, classifications)]
    # Mark hallucinations
    df['is_hallucination'] = (df['classification'].isin(['contradiction', 'neutral'])).astype(int)
    
    return df

def main():
    try:
        # Read the dataset
        print("Reading dataset...")
        df = pd.read_csv('../data/gpt5_responses.csv')
        print(f"Loaded dataset with {len(df)} rows")
    except Exception as e:
        print(f"Error reading dataset: {str(e)}")
    
    # Analyze hallucinations
    print("Analyzing hallucinations...")
    df = analyze_hallucinations(df)
    
    # Save results with hallucination annotations
    print("\nSaving results...")
    df.to_csv('../data/gpt5_responses_with_hallucinations.csv', index=False)
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
