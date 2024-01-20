# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

df_original = pd.read_csv('2023_LoL_esports.csv')
df_tratado = pd.read_csv('Dados_Tratados.csv')

colunas_interesse = ['kills', 'deaths', 'assists', 'barons', 'inhibitors','controlwardsbought', 'totalgold','minionkills','monsterkills' ]
df_original = df_original[colunas_interesse]
df_tratado = df_tratado[colunas_interesse]

num_colunas = len(colunas_interesse)
num_linhas = 1
plt.figure(figsize=(5 * num_colunas, 5 * num_linhas))

num_colunas = len(colunas_interesse)
num_linhas = 2 
plt.figure(figsize=(5 * num_colunas, 5 * num_linhas))

for i, coluna in enumerate(colunas_interesse, 1):
    plt.subplot(num_linhas, num_colunas, i)
    plt.hist(df_original[coluna], bins='auto', edgecolor='black', label='Original', alpha=0.7)
    plt.title(f'Original - {coluna.capitalize()}')
    plt.xlabel(coluna.capitalize())
    plt.ylabel('Frequencia')
    plt.legend()

    plt.subplot(num_linhas, num_colunas, i + num_colunas)
    plt.hist(df_tratado[coluna], bins='auto', edgecolor='black', label='Tratado', alpha=0.7)
    plt.title(f'Tratado - {coluna.capitalize()}')
    plt.xlabel(coluna.capitalize())
    plt.ylabel('Frequencia')
    plt.legend()

plt.tight_layout() 
plt.savefig('comparacao_histogramas.png')

pdf_filename = 'comparacao_histogramas.pdf'
c = canvas.Canvas(pdf_filename)
img_width = 500
img_height = 300
c.drawInlineImage('comparacao_histogramas.png', 50, 500, img_width, img_height)
c.showPage()
c.save()

print(f"PDF salvo com sucesso: {pdf_filename}")