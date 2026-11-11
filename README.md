# Clusterização com KMeans - Boston Housing

Projeto de agrupamento (aprendizado não supervisionado) da base **Boston Housing** usando o algoritmo **KMeans**, dividido em dois módulos: **treinamento** e **inferência**.

## Estrutura do repositório

```
├── HousingData.csv                  # Base de dados
├── housing_clusterizador.py         # Módulo de treinamento
├── housing_inferencia.py            # Módulo de inferência
├── cotovelo_housing.png             # Gráfico do método do cotovelo
├── housing_kmeans.model             # Modelo KMeans treinado
├── normalizador_housing.model       # MinMaxScaler (colunas quantitativas)
├── imputador_quant_housing.model    # Imputador de ausentes (mediana)
└── imputador_cat_housing.model      # Imputador de ausentes (moda)
```

## Requisitos

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib

```bash
pip install pandas numpy scikit-learn matplotlib
```

> Os arquivos `.model` foram salvos com `pickle` e dependem da versão do scikit-learn. Se houver erro ao carregá-los, rode o treinamento novamente para regerá-los.

## Como executar

1. **Treinar** o modelo (gera os arquivos `.model` e o gráfico do cotovelo):

   ```bash
   python housing_clusterizador.py
   ```

2. **Inferir** o grupo de uma nova residência:

   ```bash
   python housing_inferencia.py
   ```

   Para testar outra casa, edite o dicionário `nova_residencia` dentro de `housing_inferencia.py`.

## Módulo de treinamento

1. **Carga dos dados** a partir do `HousingData.csv`.
2. **Separação dos tipos de variáveis**
   - Categórica: `CHAS` (binária, 0/1), que **não** é normalizada.
   - Quantitativas: `CRIM`, `ZN`, `INDUS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`.
   - `MEDV` (preço) não entra no agrupamento, pois uma nova residência chega sem esse valor.
3. **Tratamento de valores ausentes** (o KMeans não aceita `NaN`): mediana nas quantitativas e moda na `CHAS`.
4. **Normalização** com `MinMaxScaler`, aplicada somente às variáveis quantitativas.
5. **Método do cotovelo**: cálculo da inércia (soma das distâncias quadráticas aos centroides) para k de 1 a 10, com detecção automática do cotovelo.
6. **Treinamento final** do KMeans com o k ótimo e salvamento dos modelos.

O balanceamento da base não é necessário, pois não se trata de um problema de classificação.

### Resultados do treinamento

| k | Inércia |
|---|---------|
| 1 | 387,63 |
| 2 | 217,52 |
| 3 | 169,72 |
| 4 | 148,89 |
| 5 | 128,57 |
| 6 | 111,79 |
| 7 | 97,72 |
| 8 | 89,77 |
| 9 | 82,52 |
| 10 | 75,43 |

Número ótimo de grupos pelo método do cotovelo: **k = 3**.

| Grupo | Residências |
|-------|-------------|
| 0 | 178 |
| 1 | 137 |
| 2 | 191 |

![Método do cotovelo](cotovelo_housing.png)

## Módulo de inferência

1. Recebe os dados de uma nova residência.
2. Carrega os modelos salvos (imputadores, normalizador e KMeans).
3. Trata ausentes e normaliza os dados da nova residência com o mesmo normalizador do treino.
4. Infere a qual grupo ela pertence.
5. Exibe o **centroide do grupo desnormalizado** (`inverse_transform`), ou seja, nas unidades originais da base.

Na variável `CHAS`, o valor do centroide é a proporção de casas do grupo que fazem fronteira com o rio (entre 0 e 1).

### Exemplo de saída

```
A nova residência pertence ao grupo: 2

Centroide desnormalizado do grupo 2
           valor
CRIM       0.654
ZN         1.822
INDUS     11.341
NOX        0.574
RM         6.263
AGE       82.924
DIS        3.002
RAD        4.654
TAX      330.628
PTRATIO   17.962
B        373.295
LSTAT     13.218
CHAS       0.105
```

## Dicionário das variáveis

| Variável | Descrição |
|----------|-----------|
| CRIM | Taxa de criminalidade per capita |
| ZN | Proporção de terrenos residenciais com lotes acima de 25.000 pés² |
| INDUS | Proporção de áreas comerciais não varejistas |
| CHAS | Faz fronteira com o rio Charles (1 = sim, 0 = não) |
| NOX | Concentração de óxidos nítricos |
| RM | Número médio de cômodos por residência |
| AGE | Proporção de residências construídas antes de 1940 |
| DIS | Distância ponderada até centros de emprego |
| RAD | Índice de acessibilidade a rodovias radiais |
| TAX | Taxa de imposto sobre a propriedade |
| PTRATIO | Razão aluno-professor |
| B | Índice relacionado à proporção de população negra na região |
| LSTAT | Percentual da população de baixa renda |
| MEDV | Valor mediano das residências (não usado no agrupamento) |
