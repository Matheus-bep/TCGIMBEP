# -*- coding: utf-8 -*-

from scipy.stats import shapiro
import pandas as pd

df = pd.read_csv('Dados Originais.csv')

df = df[df['position'] == 'team']

variaveis = [['side', 'kills', 'deaths', 'assists', 'firstblood', 'ckpm', 'firstdragon', 'elementaldrakes', 'elders', 'firstherald', 'heralds', 'barons', 'firsttower', 'towers', 'firstmidtower', 'firsttothreetowers', 'turretplates', 'inhibitors', 'dpm', 'damagetakenperminute', 'wardsplaced', 'wpm', 'controlwardsbought', 'visionscore', 'vspm', 'totalgold', 'minionkills', 'monsterkills', 'cspm', 'goldat10', 'xpat10', 'csat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10', 'killsat10', 'assistsat10', 'deathsat10', 'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']]

for variavel in variaveis:
    stat, p_value = shapiro(df[variavel])
    print("Estatistica de teste:", stat, "\nP-valor:", p_value)