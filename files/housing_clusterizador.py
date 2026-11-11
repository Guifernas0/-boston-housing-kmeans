import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # permite gerar o gráfico sem interface gráfica
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from pickle import dump

# ---------------------------------------------------------------
# MÓDULO DE TREINAMENTO - KMeans com a base Boston Housing
# ---------------------------------------------------------------

#1 CARREGAR OS DADOS
dados = pd.read_csv('HousingData.csv')

#2 SEPARAR OS ATRIBUTOS QUANTITATIVOS DOS CATEGÓRICOS
# CHAS é categórica binária (0/1): não pode ser normalizada como as demais.
# MEDV é o preço (alvo do dataset original); não entra no agrupamento,
# pois uma nova residência chega sem esse valor.
colunas_categoricas = ['CHAS']
colunas_quantitativas = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE',
                         'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
dados_quant = dados[colunas_quantitativas]
dados_cat = dados[colunas_categoricas]

#3 TRATAR VALORES AUSENTES (o KMeans não aceita NaN)
# Quantitativas -> mediana (robusta a outliers como CRIM) / Categórica -> moda
imputador_quant = SimpleImputer(strategy='median').fit(dados_quant)
imputador_cat = SimpleImputer(strategy='most_frequent').fit(dados_cat)
dump(imputador_quant, open('imputador_quant_housing.model', 'wb'))
dump(imputador_cat, open('imputador_cat_housing.model', 'wb'))

dados_quant = pd.DataFrame(imputador_quant.transform(dados_quant), columns=colunas_quantitativas)
dados_cat = pd.DataFrame(imputador_cat.transform(dados_cat), columns=colunas_categoricas)

#4 NORMALIZAR OS DADOS (somente os quantitativos)
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_quant)
dump(modelo_normalizador, open('normalizador_housing.model', 'wb'))
dados_quant_normalizados = pd.DataFrame(modelo_normalizador.transform(dados_quant),
                                        columns=colunas_quantitativas)

#5 JUNTAR QUANTITATIVOS NORMALIZADOS + CATEGÓRICOS (já em 0/1, mesma escala)
dados_normalizados = pd.concat([dados_quant_normalizados, dados_cat], axis=1)

#6 MÉTODO DO COTOVELO: inércia (soma das distâncias quadráticas aos centroides) para cada k
ks = list(range(1, 11))
inercias = []
for k in ks:
    kmeans_teste = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_teste.fit(dados_normalizados)
    inercias.append(kmeans_teste.inertia_)

# Detecção automática do cotovelo: ponto da curva mais distante da reta
# que liga o primeiro ao último ponto
pontos = np.column_stack([ks, inercias]).astype(float)
pontos = (pontos - pontos.min(axis=0)) / (pontos.max(axis=0) - pontos.min(axis=0))
inicio, fim = pontos[0], pontos[-1]
direcao = (fim - inicio) / np.linalg.norm(fim - inicio)
vetores = pontos - inicio
distancias = np.abs(vetores[:, 0] * direcao[1] - vetores[:, 1] * direcao[0])
k_otimo = ks[int(np.argmax(distancias))]
print('Inércias:', dict(zip(ks, [round(i, 2) for i in inercias])))
print('Número ótimo de grupos (cotovelo):', k_otimo)

# Gráfico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(ks, inercias, marker='o')
plt.axvline(k_otimo, color='red', linestyle='--', label=f'k ótimo = {k_otimo}')
plt.xlabel('Número de grupos (k)')
plt.ylabel('Inércia (soma das distâncias quadráticas)')
plt.title('Método do cotovelo - Boston Housing')
plt.legend()
plt.savefig('cotovelo_housing.png', dpi=120, bbox_inches='tight')

#7 TREINAR DEFINITIVAMENTE COM O k ÓTIMO E SALVAR
housing_kmeans = KMeans(n_clusters=k_otimo, n_init=10, random_state=42).fit(dados_normalizados)
dump(housing_kmeans, open('housing_kmeans.model', 'wb'))

#8 RESUMO DOS GRUPOS
dados['grupo'] = housing_kmeans.labels_
print('\nResidências por grupo:')
print(dados['grupo'].value_counts().sort_index())
