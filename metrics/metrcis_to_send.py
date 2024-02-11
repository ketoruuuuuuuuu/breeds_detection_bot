import pandas as pd

df = pd.read_csv('metrics/metrics.csv',index_col=0)
df.drop('Images',axis=1,inplace=True)
df_P = df.drop(['P'],axis=1)
df_P.insert(2,'P',df['P'].values)
df_P.sort_values(by=['P'],inplace=True,ascending=False)
df_P.to_markdown('metrics/metrics_p.md',index= False)

df_R = df.drop(['R'],axis=1)
df_R.insert(2,'R',df['R'].values)
df_R.sort_values(by=['R'],inplace=True,ascending=False)
df_R.to_markdown('metrics/metrics_r.md',index= False)

df_map50 = df.drop(['mAP50'],axis=1)
df_map50.insert(2,'mAP50',df['mAP50'].values)
df_map50.sort_values(by=['mAP50'],inplace=True,ascending=False)
df_map50.to_markdown('metrics/metrics_map50.md',index= False)

df_map_50_95 = df.drop(['mAP50-95'],axis=1)
df_map_50_95.insert(2,'mAP50-95',df['mAP50-95'].values)
df_map_50_95.sort_values(by=['mAP50-95'],inplace=True,ascending=False)
df_map_50_95.to_markdown('metrics/metrics_map_50_95.md',index= False)
