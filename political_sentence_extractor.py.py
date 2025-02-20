import pandas as pd
import csv

df = pd.read_csv('dev.csv', names=['text', 'label'], quoting=csv.QUOTE_NONE, sep='\t', engine='python')


df['text'] = df['text'].astype(str).str.lower()


political_keywords = ['zabe', 'jam’iyya', 'shugaba', 'majalisar', 'minista', 'doka', 'gwamnati', 'ra’ayi', 
                      'zaben', 'siyasa', 'dan siyasa', 'shugaban kasa', 'election', 'government', 'president', 
                      'senator', 'democracy', 'APC', 'PDP', 'baba', 'buhari', 'najeriya', 'sanata', 'gomnati', 
                      'gomna', 'ganduje']


political_df = df[df['text'].str.contains('|'.join(political_keywords), case=False, na=False)]


political_df.to_csv('political_sentences3.csv', index=False)

print("Political sentences have been successfully saved to 'political_sentence31.csv'.")
