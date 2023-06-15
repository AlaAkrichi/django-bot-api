import pandas as pd
import sklearn


# préparation des données corpus-----------------------------------------------------------------------------------------

df = pd.read_excel('./data.xlsx', engine='openpyxl')

df = df.dropna()
df=sklearn.utils.shuffle(df)
data = df[['label', 'text']].rename(columns={"label": "label", "text": "text"})
data['label'] = '_label_' + data['label'].astype(str)

data.iloc[0:int(len(data)*0.8)].to_csv('data1/train.csv', sep='#', index=False, header=False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('data1/test.csv', sep='#', index=False, header=False)
data.iloc[int(len(data)*0.9):].to_csv('data1/dev.csv', sep='#', index=False, header=False)
print(data.info())
print(data.head())