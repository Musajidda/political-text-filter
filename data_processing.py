import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import re

# Load labeled dataset
df = pd.read_csv('labeled_hausa_dataset.csv')

# Convert to lowercase
df['text'] = df['text'].str.lower()

# Remove punctuation and special characters
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Define Hausa stopwords manually
hausa_stopwords = set([
    'da', 'ne', 'ce', 'a', 'na', 'ina', 'zuwa', 'ba', 'kai', 'ku', 'mu',
    'shi', 'ita', 'su', 'wanda', 'wacce', 'wadanda', 'amma', 'ko', 'saboda',
    'kuma', 'haka', 'yana', 'wannan', 'wancan', 'idan', 'sai', 'tun', 'lokaci','user'
])

# Remove Hausa stopwords
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in hausa_stopwords]))

# Apply stemming (if needed)
stemmer = PorterStemmer()
df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save preprocessed data
train_df.to_csv('train_hausa_dataset.csv', index=False)
test_df.to_csv('test_hausa_dataset.csv', index=False)

print("Data preprocessing completed and datasets saved.")
