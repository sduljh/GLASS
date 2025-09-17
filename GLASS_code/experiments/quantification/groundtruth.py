import pandas as pd

df = pd.read_csv('ONT.simulated.read_to_isoform.tsv',
                 sep='\t', header=None, names=['read_id', 'transcript_id'])

counts = df['transcript_id'].value_counts().reset_index()
counts.columns = ['transcript_id', 'count']

counts.to_csv('transcript_expression.tsv', sep='\t', index=False)
