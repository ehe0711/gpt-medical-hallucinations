import pandas as pd

df = pd.read_csv("C:/Users/tianc/OneDrive/Desktop/Projects/llm_hallucinations/data/dataset.csv")

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check for duplicate questions (exact word-for-word matching)
duplicates = df[df['question'].duplicated(keep=False)]['question'].unique()

print(f"\nDuplicate questions found: {len(duplicates)}")
if len(duplicates) > 0:
    print("\nList of duplicate questions with counts:")
    for i, question in enumerate(duplicates, 1):
        count = df[df['question'] == question].shape[0]
        print(f"{i}. {question} (appears {count} times)")
else:
    print("No duplicate questions found.")

# Remove duplicate questions from the dataset
print("Original dataset shape:", df.shape)

# Remove duplicates based on the 'question' column, keeping the first occurrence
df_no_duplicates = df.drop_duplicates(subset=['question'], keep='first')

print("Dataset shape after removing duplicates:", df_no_duplicates.shape)
print(f"Removed {len(df) - len(df_no_duplicates)} duplicate questions")
print(f"Reduction: {((len(df) - len(df_no_duplicates)) / len(df)) * 100:.2f}%")

# Update the original dataframe if you want to keep the changes
df = df_no_duplicates

print("\nDataset updated. New shape:", df.shape)

# Save the cleaned dataset without duplicates
output_path = "C:/Users/tianc/OneDrive/Desktop/Projects/llm_hallucinations/data/dataset_minus_duplicates.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")
print("This dataset contains no duplicate questions.")