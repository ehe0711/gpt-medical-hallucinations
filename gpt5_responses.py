import os
import pandas as pd
from openai import OpenAI
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force output to show immediately
sys.stdout.reconfigure(line_buffering=True)

def print_flush(*args, **kwargs):
    """Print and flush immediately"""
    print(*args, **kwargs)
    sys.stdout.flush()

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_API_KEY")  # Replace with your actual API key or use environment variable

# Settings
BATCH_SIZE = 25  # Medium batch size for balanced processing
START_ROW = 1920  # Resume from index 1920
MAX_WORKERS = 22  # Slightly increased worker count
TIMEOUT = 30  # Timeout for API calls in seconds
RETRY_DELAY = 1  # Delay between retries in seconds

# Medical expert response prompt
MEDICAL_EXPERT_PROMPT = """You are a highly knowledgeable medical expert providing concise, accurate information. Answer the following medical question in a single, focused paragraph. Give a complete answer that covers all important aspects of the specific question asked, while being as concise as possible. Use proper medical terminology but keep the response clear and accessible. Every word should add value - include all crucial information related to the question, but avoid straying beyond what was asked. Keep responses under 100 words while maximizing information density.

Question: {question}"""

# Confidence scoring prompt
CONFIDENCE_PROMPT = """Rate your confidence in the accuracy and completeness of this response as a precise number between 0.00 and 1.00. Consider subtle differences in accuracy and completeness when choosing between similar numbers (like 0.87 vs 0.86). Be exact in your assessment - each hundredth decimal point matters.

Return only the number with exactly two decimal places, nothing else.

Question: {question}
Response: {response}"""

def get_medical_response(question):
    """Get medical response from GPT with optimized error handling and timeouts"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a specialized medical expert focusing on precise, evidence-based responses."},
                    {"role": "user", "content": MEDICAL_EXPERT_PROMPT.format(question=question)}
                ],
                timeout=TIMEOUT
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower():
                # Rate limit hit - wait longer
                time.sleep(RETRY_DELAY * (attempt + 2))
            elif attempt < max_retries - 1:
                # Other error - standard retry
                time.sleep(RETRY_DELAY)
            else:
                print_flush(f"Error getting medical response after {max_retries} attempts: {e}")
                return None

def get_confidence_score(question, response):
    """Get precise confidence score with optimized error handling and timeouts"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            conf_response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a medical accuracy assessor. Be extremely precise with confidence scores."},
                    {"role": "user", "content": CONFIDENCE_PROMPT.format(question=question, response=response)}
                ],
                timeout=TIMEOUT
            )
            
            score_text = conf_response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                if 0 <= score <= 1:  # Validate score is in valid range
                    return score
                print_flush(f"Invalid confidence score range: {score}")
                return None
            except ValueError:
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                print_flush(f"Could not convert confidence score to float: {score_text}")
                return None
                
        except Exception as e:
            if "rate limit" in str(e).lower():
                # Rate limit hit - wait longer
                time.sleep(RETRY_DELAY * (attempt + 2))
            elif attempt < max_retries - 1:
                # Other error - standard retry
                time.sleep(RETRY_DELAY)
            else:
                print_flush(f"Error getting confidence score after {max_retries} attempts: {e}")
                return None

def process_single_question(args):
    """Process a single question with optimized API calls"""
    idx, question = args
    max_retries = 2
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = get_medical_response(question)
            if not response:
                continue
                
            confidence = get_confidence_score(question, response)
            if confidence is not None:
                print_flush(f"✓ Q{idx} ({len(response)} chars, conf:{confidence})")
                return {
                    'index': idx,
                    'response': response,
                    'confidence': confidence
                }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print_flush(f"× Q{idx} failed: {str(e)[:100]}")
            
    return None

def process_batch(df, start_idx, batch_size):
    """Process a batch of questions with optimized parallel execution"""
    end_idx = min(start_idx + batch_size, len(df))
    batch = df.iloc[start_idx:end_idx]
    
    print_flush(f"\n[Batch {start_idx}-{end_idx}]")
    
    # Pre-fetch questions and prepare futures
    questions = [(idx, row['question']) for idx, row in batch.iterrows()]
    results = []
    
    # Process with aggressive parallelization
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_question, q) for q in questions]
        
        # Handle results as they complete
        for future in as_completed(futures):
            if (result := future.result()):
                results.append(result)
                
                # Save progress every 2 completed items
                if len(results) % 2 == 0:
                    for resp in results[-2:]:  # Save last 2 results
                        df.at[resp['index'], 'llm_response'] = resp['response']
                        df.at[resp['index'], 'llm_confidence'] = resp['confidence']
                    
                    try:
                        temp_path = Path(df.attrs.get('filepath', 'output.csv'))
                        df.to_csv(temp_path, index=False)
                    except Exception as e:
                        print_flush(f"Interim save failed: {str(e)[:50]}")
    
    return results

def process_csv():
    """Process the CSV file in batches"""
    print_flush("\nStarting medical response processing...")
    
    try:
        # Get absolute path to CSV
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data"
        csv_path = data_dir / "gpt5_responses.csv"
        
        print_flush(f"\nReading from: {csv_path}")
        
        # Check file exists
        if not csv_path.exists():
            print_flush(f"Error: File not found: {csv_path}")
            return
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print_flush(f"Loaded {len(df)} rows")
        
        if START_ROW >= len(df):
            print_flush(f"Error: Start row {START_ROW} exceeds file length {len(df)}")
            return
        
        # Process in batches
        current_row = START_ROW
        while current_row < len(df):
            print_flush(f"\n{'='*50}")
            print_flush(f"Starting batch at row {current_row}")
            
            # Process batch
            batch_responses = process_batch(df, current_row, BATCH_SIZE)
            
            # Update dataframe with responses
            for resp in batch_responses:
                df.at[resp['index'], 'gpt5_response'] = resp['response']
                df.at[resp['index'], 'gpt5_confidence'] = resp['confidence']
                
            # Save progress with retries
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Try to save with a temporary file first
                    temp_path = data_dir / f"temp_{int(time.time())}.csv"
                    df.to_csv(temp_path, index=False)
                    
                    # If successful, replace the original file
                    if temp_path.exists():
                        # On Windows, we need to remove the target file first
                        if csv_path.exists():
                            csv_path.unlink()
                        temp_path.replace(csv_path)
                        print_flush("Saved progress to CSV")
                        break
                except Exception as e:
                    print_flush(f"Save attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        print_flush(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print_flush("All save attempts failed. Please ensure the file is not open in another program.")
            
            # Move to next batch
            current_row += BATCH_SIZE
            print_flush(f"Progress: {min(current_row, len(df))}/{len(df)} rows")
            print_flush(f"{'='*50}\n")
        
        print_flush("\nProcessing complete!")
        
    except Exception as e:
        print_flush(f"Error processing CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print_flush("Starting script...")
    process_csv()
