# Utilizando o KNN para Identificar Dígitos escritos a Mão

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd

#### A base de digitos
digits = datasets.load_digits()

## Descrição sobre a base de dados
print(digits.DESCR)

#### Visualizando os valores de dados
digits.images 

#### Visualizando os valores de classes
digits.target_names 

#### Visualizando as imagens e classes
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
    
## convertendo os dados em Dataframe
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
classe = digits.target 

dataset = pd.DataFrame(data)
dataset['classe'] = classe 

''' Cada valor mostrado nas colunas é a intencidade de pixels'''
dataset.head() 

## Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('classe', axis=1), dataset['classe'], test_size=0.3)

#### Verificando a forma dos dados
X_train.shape, X_test.shape
y_train.shape, y_test.shape

## Instânciando o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=3) # p=2 e metric='minkowski' indica a distancia euclidiana

### Treinando o algoritmo
knn.fit(X_train, y_train)

### Predizendo os novos pontos
resultado = knn.predict(X_test)

# Técnica de Validação
### Metricas de classificação
print(metrics.classification_report(y_test, resultado))

### Matriz de Confusão
print(pd.crosstab(y_test, resultado, rownames=['Real'], colnames=['         Predito'], margins=True))

## Cros Validation
scores = cross_val_score(knn, dataset.drop('classe', axis=1), dataset['classe'], cv=5)
scores
