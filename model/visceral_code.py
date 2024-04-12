import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'model\visceral_fat.csv')
df['current gestational age'] = [int(age[0])*7 + int(age[1]) for age in df['current gestational age'].str.split(',')]
df['gestational age at birth'] = [int(age[0])*7 + int(age[1]) for age in df['gestational age at birth'].str.split(',')]

df['gestational dm'] = df['gestational dm'].astype('bool')
df['diabetes mellitus'] = df['diabetes mellitus'].astype('bool')
df['type of delivery'] = df['type of delivery'].astype('bool')
df['ethnicity'] = df['ethnicity'].astype('bool')

plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True, cmap="crest")
plt.show()