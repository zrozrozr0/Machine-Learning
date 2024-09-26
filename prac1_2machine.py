import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Leer el dataset y convertirlo en un DataFrame
dataset = pd.read_csv("peleteria.csv")

# Revolver las instancias del DataFrame
dataset = shuffle(dataset, random_state=42)

# Imprimir el DataFrame completo
print("DataFrame completo:")
print(dataset)

# Separar las características del DataFrame (todas las columnas excepto la última)
X = dataset.iloc[:, :-1]

# Imprimir las características
print("\nCaracterísticas:")
print(X)

# Separar las etiquetas del DataFrame (última columna)
y = dataset.iloc[:, -1]

# Imprimir las etiquetas
print("\nEtiquetas:")
print(y)

# Separar el dataset en un subconjunto de entrenamiento (90%) y un subconjunto de prueba (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Imprimir los subconjuntos de entrenamiento y prueba
print("\nSubconjunto de entrenamiento:")
print(X_train)
print(y_train)

print("\nSubconjunto de prueba:")
print(X_test)
print(y_test)
