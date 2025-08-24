#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Creation Script

This script extracts prompts where both GPT-4 and GPT-5 hallucinated and creates
a new dataset of challenging medical questions in the same format as the original dataset.
"""

import pandas as pd
import os
from pathlib import Path

def create_hallucination_dataset():
    """
    Create a dataset of questions that both GPT-4 and GPT-5 hallucinated on.
    
    Returns:
        pd.DataFrame: The dataset of challenging questions
    """
    print("Loading datasets...")
    
    # Define data paths
    data_dir = Path("../data")
    gpt4_file = data_dir / "gpt4_responses_with_hallucinations.csv"
    gpt5_file = data_dir / "gpt5_responses_with_hallucinations.csv"
    
    # Load data
    df_gpt4 = pd.read_csv(gpt4_file)
    df_gpt5 = pd.read_csv(gpt5_file)
    
    print(f"Loaded {len(df_gpt4)} GPT-4 responses and {len(df_gpt5)} GPT-5 responses")
    
    # Get questions where both models hallucinated (is_hallucination = 1)
    hallucinated_gpt4 = set(df_gpt4[df_gpt4['is_hallucination'] == 1]['question'])
    hallucinated_gpt5 = set(df_gpt5[df_gpt5['is_hallucination'] == 1]['question'])
    
    # Find the intersection of questions that both models hallucinated on
    common_hallucinations = hallucinated_gpt4.intersection(hallucinated_gpt5)
    
    print(f"Found {len(common_hallucinations)} questions that both models hallucinated on")
    
    # Create a new dataframe with only the original columns from dataset_minus_duplicates.csv
    # Get the data from either GPT-4 or GPT-5 dataframe (they should have the same source data)
    challenging_dataset = df_gpt4[df_gpt4['question'].isin(common_hallucinations)][['question', 'answer', 'source', 'focus_area']]
    
    return challenging_dataset

def main():
    """
    Main function to create and save the dataset of challenging questions.
    """
    output_dir = Path("../data")
    output_file = output_dir / "challenging_medical_dataset.csv"
    
    print("Creating dataset of challenging medical questions...")
    challenging_dataset = create_hallucination_dataset()
    
    # Save the dataset
    challenging_dataset.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Total questions: {len(challenging_dataset)}")

if __name__ == "__main__":
    main()
