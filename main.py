import json
import pandas as pd
# import os
# import webbrowser
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# import geopandas as gpd
# from sklearn.cluster import KMeans
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# import tensorflow as tf
# from tensorflow import keras
# import time
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Dropout

# Pandas
df = pd.read_csv('dataset_8_urbanshield.csv')

# Crie um dicionário que mapeia os nomes atuais das colunas para os novos nomes desejados
novo_nome_colunas = {
    'Rua': 'rua',
    'Presença': 'presenca',
    'Luminosidade': 'luminosidade',
    'Ruído': 'ruido',
    'Denúncias_Recentes': 'denuncias_recentes',
    'Classificação': 'classificacao'
}

# Use o método rename para renomear as colunas
df.rename(columns=novo_nome_colunas, inplace=True)

df.head()

# Mapeie os valores numéricos para categorias
mapeamento_categorias = {
    0: 'Segura',
    1: 'Perigosa',
    2: 'Muito Perigosa',
}

# Codifique a variável alvo (Classificação)
label_encoder = LabelEncoder()
df['classificacao'] = label_encoder.fit_transform(df['classificacao'])

# Divida os dados em conjuntos de treinamento e teste
X = df[['presenca', 'luminosidade', 'ruido', 'denuncias_recentes']]
y = df['classificacao']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Faça previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avalie o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Funções de login
def carregar_contas():
    try:
        with open('contas.json') as arquivo:
            return json.load(arquivo)
    except FileNotFoundError:
        return {}

def salvar_contas(contas):
    with open('contas.json', 'w') as arquivo:
        json.dump(contas, arquivo)

contas = carregar_contas()

def criar_conta():
    nome = input("Digite o nome de usuário: ")
    senha = input("Digite a senha: ")

    if nome in contas:
        print("Nome de usuário já existe!")
        return

    contas[nome] = senha

    print("Conta criada com sucesso!")

    salvar_contas(contas)

def fazer_login():
    nome = input("Digite o nome de usuário: ")
    senha = input("Digite a senha: ")

    if nome in contas and contas[nome] == senha:
        print("Login bem-sucedido!")
        segundo_menu(nome)
    else:
        print("Nome de usuário ou senha incorretos.")

def menu():
    continuar = True

    while continuar:
        print("\nMenu:")
        print("1. Criar Conta")
        print("2. Fazer Login")
        print("3. Sair")

        try:
            escolha = int(input("Escolha uma opção: "))
        except ValueError:
            print("Opção inválida. Digite apenas números.")
            continue

        if escolha == 1:
            criar_conta()
        elif escolha == 2:
            fazer_login()
        elif escolha == 3:
            continuar = False
        else:
            print("Opção inválida. Escolha novamente.")

def segundo_menu(nome):
    sair = False

    while not sair:
        print("\nSegundo Menu:")
        print("1. Realizar uma previsão")
        print("2. Sair da conta")

        try:
            escolha = int(input("Escolha uma opção: "))
        except ValueError:
            print("Opção inválida. Digite apenas números.")
            continue

        if escolha == 1:
            # Solicite ao usuário que insira os dados da rua
            presenca = float(input("Informe a presença de pessoas (0 a 100): "))
            luminosidade = float(input("Informe a luminosidade (0 a 100): "))
            ruido = float(input("Informe o nível de ruído: "))
            denuncias_recentes = int(input("Informe o número de denúncias recentes: "))

            novo_dado = [[presenca, luminosidade, ruido, denuncias_recentes]]

            # Faça a previsão
            previsao = model.predict(novo_dado)

            # Use a função inversa para obter a classe original
            classe_original = label_encoder.inverse_transform(previsao)

            print(f"Classificação da rua: {classe_original[0]}")
        elif escolha == 2:
            print("Saindo da conta...")
            sair = True
        else:
            print("Opção inválida. Escolha novamente.")

menu()
