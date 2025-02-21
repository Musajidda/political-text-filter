import pandas as pd

# Load datasets
positive_df = pd.read_csv('Positive_sentences.csv')
neutral_df = pd.read_csv('Neutral_sentences.csv')
negative_df = pd.read_csv('Negative_sentences.csv')

# Assign labels
positive_df['label'] = 'Positive'
neutral_df['label'] = 'Neutral'
negative_df['label'] = 'Negative'

# Combine datasets
merged_df = pd.concat([positive_df, neutral_df, negative_df], ignore_index=True)

# Save the labeled dataset
merged_df.to_csv('labeled_hausa_dataset.csv', index=False)

print("Labeled dataset created and saved successfully.")