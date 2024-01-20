import pandas as pd

# Substitua 'seu_arquivo.csv' pelo caminho do seu arquivo CSV
caminho_arquivo = 'Dados Originais.csv'

# Carregando o arquivo CSV em um DataFrame do pandas
dados = pd.read_csv(caminho_arquivo)

# Contando valores ausentes em cada coluna
contagem_missings_por_coluna = dados.isnull().sum()

# Mostrando a contagem de missings por coluna
for coluna, quantidade in contagem_missings_por_coluna.items():
    print(f'Coluna {coluna}: {quantidade} missings')
