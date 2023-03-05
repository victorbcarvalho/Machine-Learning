import pandas  as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


## Carregando a base de dados
iris = pd.read_csv('iris.csv')

## Verificando os atributos
iris.head()

iris.describe()

## Dividendo os dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(iris.drop('Species', axis=1), iris['Species'], test_size=0.2)

## verificando a forma dos dados
X_train.shape, x_test.shape 
Y_train.shape, y_test.shape 

## instânciando o algoritmo KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

## Treinando o algoritmo
knn.fit(X_train, Y_train)

## Executando o KNN com o conjunto de teste
resultado = knn.predict(x_test)
resultado

### Executando novas amostras
test = np.array([[5.1, 3.5, 1.4, 0.2]])
knn.predict(test), knn.predict_proba(test)

# Técnica de Validação
## Matriz de Confusão

print(pd.crosstab(y_test, resultado, rownames=['Real'], colnames=['          Predito'], margins=True))

## metricas de classificação
from sklearn import metrics
print(metrics.classification_report(y_test, resultado, target_names=iris['Species'].unique()))