import pandas as pd
from pickle import load

# ---------------------------------------------------------------
# MÓDULO DE INFERÊNCIA - a qual grupo pertence uma nova residência?
# ---------------------------------------------------------------
colunas_quantitativas = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE',
                         'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
colunas_categoricas = ['CHAS']

#1 RECEBER OS DADOS DA NOVA RESIDÊNCIA (MEDV não é necessário)
nova_residencia = pd.DataFrame([{
    'CRIM': 0.08, 'ZN': 0.0, 'INDUS': 8.5, 'CHAS': 0, 'NOX': 0.52,
    'RM': 6.4, 'AGE': 55.0, 'DIS': 4.2, 'RAD': 5, 'TAX': 300,
    'PTRATIO': 17.5, 'B': 390.0, 'LSTAT': 9.5
}])

#2 CARREGAR OS MODELOS SALVOS NO TREINAMENTO
imputador_quant = load(open('imputador_quant_housing.model', 'rb'))
imputador_cat = load(open('imputador_cat_housing.model', 'rb'))
normalizador = load(open('normalizador_housing.model', 'rb'))
housing_kmeans = load(open('housing_kmeans.model', 'rb'))

#3 TRATAR AUSENTES E NORMALIZAR (somente quantitativos; CHAS permanece 0/1)
quant = pd.DataFrame(imputador_quant.transform(nova_residencia[colunas_quantitativas]),
                     columns=colunas_quantitativas)
cat = pd.DataFrame(imputador_cat.transform(nova_residencia[colunas_categoricas]),
                   columns=colunas_categoricas)
quant_normalizado = pd.DataFrame(normalizador.transform(quant), columns=colunas_quantitativas)
nova_residencia_normalizada = pd.concat([quant_normalizado, cat], axis=1)

#4 INFERIR O GRUPO
grupo = housing_kmeans.predict(nova_residencia_normalizada)[0]
print('A nova residência pertence ao grupo:', grupo)

#5 EXIBIR O CENTROIDE DO GRUPO DESNORMALIZADO
centroide = housing_kmeans.cluster_centers_[grupo]
centroide_quant = centroide[:len(colunas_quantitativas)].reshape(1, -1)
centroide_quant = normalizador.inverse_transform(centroide_quant)
centroide_desnormalizado = pd.DataFrame(centroide_quant, columns=colunas_quantitativas)
# CHAS não foi normalizada: o valor do centroide é a proporção de casas na margem do rio (0 a 1)
centroide_desnormalizado['CHAS'] = centroide[len(colunas_quantitativas):]

print('\nCentroide desnormalizado do grupo', grupo)
print(centroide_desnormalizado.round(3).T.rename(columns={0: 'valor'}))
