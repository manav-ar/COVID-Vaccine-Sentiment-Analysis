import pandas as pd
#df = pd.read_json (r'C:\twitter\covid.json')
#df.to_csv (r'C:\twitter\covid.csv', lines=True)

with open('covid4.json', 'rb') as f:
    data = f.readlines()

data = map(lambda x: x.rstrip(), data)


data_json_str = b"[" + b','.join(data) + b"]"

data_df = pd.read_json(data_json_str)

data_df.to_csv (r'C:\twitter\covid4.csv')
