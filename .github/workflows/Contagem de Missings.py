import pandas as pd

caminho_arquivo = 'Dados Originais.csv'

dados = pd.read_csv(caminho_arquivo)

contagem_missings_por_coluna = dados.isnull().sum()

for coluna, quantidade in contagem_missings_por_coluna.items():
    print(f'Coluna {coluna}: {quantidade} missings')
