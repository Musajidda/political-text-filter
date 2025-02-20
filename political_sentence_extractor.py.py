import pandas as pd
import csv

# Load the dataset with defined column names
df = pd.read_csv('dev.csv', names=['text', 'label'], quoting=csv.QUOTE_NONE, sep='\t', engine='python')

# Convert text to lowercase for consistent searching
df['text'] = df['text'].astype(str).str.lower()

# Define keywords related to politics
political_keywords = ['zabe', 'jam’iyya', 'shugaba', 'majalisar', 'minista', 'doka', 'gwamnati', 'ra’ayi', 
                      'zaben', 'siyasa', 'dan siyasa', 'shugaban kasa', 'election', 'government', 'president', 
                      'senator', 'democracy', 'APC', 'PDP', 'baba', 'buhari', 'najeriya', 'sanata', 'gomnati', 
                      'gomna', 'ganduje']

# Filter political-related sentences
political_df = df[df['text'].str.contains('|'.join(political_keywords), case=False, na=False)]

# Save political-related sentences to a new CSV file
political_df.to_csv('political_sentences3.csv', index=False)

print("Political sentences have been successfully saved to 'political_sentence31.csv'.")
