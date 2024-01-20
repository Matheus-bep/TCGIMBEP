# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv('2023_LoL_esports.csv')

df = df[df['position'] == 'team']

df['gamelength'] = df['gamelength'].apply(lambda x: float(x) / 100)

df['league'] = df['league'].map(lambda x: 'Outras' if x not in ['LPL', 'LCK', 'CBLOL', 'MSI', 'WLDs'] else x)

mediana_kills = df.groupby('gamelength')['kills'].median()
mediana_deaths = df.groupby('gamelength')['deaths'].median()
mediana_assists = df.groupby('gamelength')['assists'].median()
mediana_barons = df.groupby('gamelength')['barons'].median()
mediana_inhibitors = df.groupby('gamelength')['inhibitors'].median()
mediana_controlwardsbought = df.groupby('gamelength')['controlwardsbought'].median()
mediana_totalgold = df.groupby('gamelength')['totalgold'].median()
mediana_minionkills = df.groupby('gamelength')['minionkills'].median()
mediana_monsterkills = df.groupby('gamelength')['monsterkills'].median()

df['kills'] = df.apply(lambda row: row['kills'] / mediana_kills[row['gamelength']], axis=1)
df['deaths'] = df.apply(lambda row: row['deaths'] / mediana_deaths[row['gamelength']], axis=1)
df['assists'] = df.apply(lambda row: row['assists'] / mediana_assists[row['gamelength']], axis=1)
df['barons'] = df.apply(lambda row: row['barons'] / mediana_barons[row['gamelength']] if mediana_barons[row['gamelength']] != 0 else 0, axis=1)
df['inhibitors'] = df.apply(lambda row: row['inhibitors'] / mediana_inhibitors[row['gamelength']] if mediana_inhibitors[row['gamelength']] != 0 else 0, axis=1)
df['controlwardsbought'] = df.apply(lambda row: row['controlwardsbought'] / mediana_controlwardsbought[row['gamelength']], axis=1)
df['totalgold'] = df.apply(lambda row: row['totalgold'] / mediana_totalgold[row['gamelength']], axis=1)
df['minionkills'] = df.apply(lambda row: row['minionkills'] / mediana_minionkills[row['gamelength']], axis=1)
df['monsterkills'] = df.apply(lambda row: row['monsterkills'] / mediana_monsterkills[row['gamelength']], axis=1)

dragon_columns = ['infernals', 'mountains', 'clouds', 'oceans', 'chemtechs', 'hextechs', 'dragons (type unknown)']

df['dragons'] = df[dragon_columns].apply(lambda x: ', '.join(x.index[x == 1]), axis=1)

df.drop(columns=dragon_columns, inplace=True)

df.to_csv('Dados_Tratados.csv', index=False)
