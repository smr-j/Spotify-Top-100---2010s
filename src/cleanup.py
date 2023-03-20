import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import seaborn as sb

fpath = "data/spotify_top_100_2010_2019.csv"
df = pd.read_csv(fpath)

df = df.dropna()

col_val = []
for x in range(len(df['genres'])):
  temp = []
  temp =(df['genres'][x].lstrip("['").rstrip("'])").split("', '"))
  col_val.append(temp[0])
df['top genre'] = col_val
del df['genres']

#limit the number of genres
df['top genre'] = df['top genre'].apply(lambda x: 'hip hop' if 'hip hop' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'hip hop' if 'rap' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'hip hop' if 'afroswing' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'hip hop' if 'g funk' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'hip hop' if 'drill' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'latin' if 'latin' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'latin' if 'sertanejo' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'latin' if 'reggaeton' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'pop' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'idol' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'talent' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'boy' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'wave' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'alt z' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'complex' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'hollywood' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'adult standards' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'reggae fusion' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'pop' if 'glee' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'rock' if 'rock' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'rock' if 'metal' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'rock' if 'emo' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'indie' if 'indie' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'indie' if 'shoe' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'indie' if 'downtem' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'indie' if 'psych' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'indie' if 'stomp' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'r&b' if 'r&b' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'r&b' if 'soul' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'r&b' if 'afrofut' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'house' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'dance' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'lilith' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'big room' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'grime' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'edm' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'techno' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'basshall' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'clubbing' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'bro' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'electro' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'aussietr' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'comic' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'new french' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'trance' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'dance' if 'covertronica' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'singer-songwriter' if 'mellow' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'singer-songwriter' if 'songwriter' in x else x)
df['top genre'] = df['top genre'].apply(lambda x: 'contemporary country' if 'black americana' in x else x)

#print(len(df['top genre'].unique()))


