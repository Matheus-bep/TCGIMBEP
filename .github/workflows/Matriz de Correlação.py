# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 


df_original = pd.read_csv('Dados Originais.csv')
df_tratado = pd.read_csv('Dados_Tratados_1212.csv')

df_original['side'], _ = pd.factorize(df_original['side'])
df_original['league'], _ = pd.factorize(df_original['league'])

df_tratado['league'], _ = pd.factorize(df_tratado['league'])
df_tratado['dragons'], _ = pd.factorize(df_tratado['dragons'])


y_result_original = df_original['result']
X_result_original = df_original[['side', 'kills', 'deaths', 'assists', 'firstblood', 'ckpm', 'firstdragon', 'elementaldrakes', 'elders', 'firstherald', 'heralds', 'barons', 'firsttower', 'towers', 'firstmidtower', 'firsttothreetowers', 'turretplates', 'inhibitors', 'dpm', 'damagetakenperminute', 'wardsplaced', 'wpm', 'controlwardsbought', 'visionscore', 'vspm', 'totalgold', 'minionkills', 'monsterkills', 'cspm', 'goldat10', 'xpat10', 'csat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10', 'killsat10', 'assistsat10', 'deathsat10', 'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']]  # Substitua com as suas variaveis

X_result_original = X_result_original.fillna(X_result_original.median())

X_train_result_original, X_test_result_original, y_train_result_original, y_test_result_original = train_test_split(X_result_original, y_result_original, test_size=0.2, random_state=42)

correlation_matrix_result_original = df_original.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_result_original, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlacao - Result Original')
plt.show()


rf_result_original = RandomForestClassifier()
rf_result_original.fit(X_train_result_original, y_train_result_original)

predictions_result_original = rf_result_original.predict(X_test_result_original)

accuracy_result_original = accuracy_score(y_test_result_original, predictions_result_original)
print("Accuracy Result Original:", accuracy_result_original)


y_result_tratado = df_tratado['result']
X_result_tratado = df_tratado.drop(['result', 'league'], axis=1)

X_result_tratado = X_result_tratado.fillna(X_result_tratado.median())

X_train_result_tratado, X_test_result_tratado, y_train_result_tratado, y_test_result_tratado = train_test_split(X_result_tratado, y_result_tratado, test_size=0.2, random_state=42)

correlation_matrix_result_tratado = df_tratado.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_result_tratado, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlacao - Result Tratado')
plt.show()

rf_result_tratado = RandomForestClassifier()
rf_result_tratado.fit(X_train_result_tratado, y_train_result_tratado)

predictions_result_tratado = rf_result_tratado.predict(X_test_result_tratado)

accuracy_result_tratado = accuracy_score(y_test_result_tratado, predictions_result_tratado)
print("Accuracy Result Tratado:", accuracy_result_tratado)


y_league_original = df_original['league']
X_league_original = df_original.drop(['result', 'league'], axis=1)

X_league_original = X_league_original.fillna(X_league_original.median())

X_train_league_original, X_test_league_original, y_train_league_original, y_test_league_original = train_test_split(X_league_original, y_league_original, test_size=0.2, random_state=1)

correlation_matrix_league_original = df_original.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_league_original, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlacao - League Original')
plt.show()

rf_league_original = RandomForestClassifier()
rf_league_original.fit(X_train_league_original, y_train_league_original)

predictions_league_original = rf_league_original.predict(X_test_league_original)

accuracy_league_original = accuracy_score(y_test_league_original, predictions_league_original)
print("Accuracy League Original:", accuracy_league_original)


y_league_tratado = df_tratado['league']
X_league_tratado = df_tratado.drop(['result', 'league'], axis=1)

X_league_tratado = X_league_tratado.fillna(X_league_tratado.median())

X_train_league_tratado, X_test_league_tratado, y_train_league_tratado, y_test_league_tratado = train_test_split(X_league_tratado, y_league_tratado, test_size=0.2, random_state=1)

correlation_matrix_league_tratado = df_tratado.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_league_tratado, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlacao - League Tratado')
plt.show()
rf_league_tratado = RandomForestClassifier()
rf_league_tratado.fit(X_train_league_tratado, y_train_league_tratado)

predictions_league_tratado = rf_league_tratado.predict(X_test_league_tratado)

accuracy_league_tratado = accuracy_score(y_test_league_tratado, predictions_league_tratado)
print("Accuracy League Tratado:", accuracy_league_tratado)
