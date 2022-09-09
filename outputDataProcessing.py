import os

import pandas as pd

inputFolder = r'output'

all_files = []
for root, dirs, files in os.walk(inputFolder):
    for name in files:
        filepath = root + os.sep + name
        if name.startswith('summary') and 'invalid' not in filepath and 'error' not in filepath:
            print(filepath)
            all_files.append(filepath)


dfs = [
    pd.read_csv(f, header=0, encoding='latin1') for f in all_files
]

for i in range(len(dfs)):
    dfs[i]['index'].values[:] = i

df = pd.concat(dfs)

df.rename(columns={'id_name':'Name_ID'}, inplace=True)
df.rename(columns={'index':'Person'}, inplace=True)
df = df.rename(columns=lambda c: '' if c.startswith('Unnamed') else c)
df.drop('age', axis=1, inplace=True)

df.to_csv('output_summary_20220909.csv')
