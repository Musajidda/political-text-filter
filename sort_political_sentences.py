import pandas as pd
import os


df = pd.read_csv('labeled_hausa_dataset.csv')


df['label'] = df['label'].str.lower()


output_dirs = ['Positive', 'Negative', 'Neutral']

for folder in output_dirs:
    os.makedirs(folder, exist_ok=True)


for label in output_dirs:
    filtered_df = df[df['label'] == label.lower()]
    filtered_df.to_csv(f'{label}/{label}_sentences.csv', index=False)

print("Sentences sorted into Positive, Negative, and Neutral folders successfully.")
