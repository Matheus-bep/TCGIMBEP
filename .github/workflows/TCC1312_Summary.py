# -*- coding: utf-8 -*-

import pandas as pd
from reportlab.pdfgen import canvas

df_original = pd.read_csv('Dados Tratados.csv')
df_tratado = pd.read_csv('Dados Originais.csv')

# Sumário - Dataset Original
summary_original = df_original.describe()

# Sumário - Dataset Tratado
summary_tratado = df_tratado.describe()

summary_original.to_csv('summary_original.csv', index=False)
summary_tratado.to_csv('summary_tratado.csv', index=False)

print("Sumario do Dataset Original:")
print(summary_original)

print("\nSumario do Dataset Tratado:")
print(summary_tratado)