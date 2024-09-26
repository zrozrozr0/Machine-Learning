import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Separar iris
iris = pd.read_csv('iris.csv')
especies = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

df_setosa = iris.copy()
df_setosa['species'] = df_setosa['species'].where(df_setosa['species'] == 'Iris-setosa', 'No-setosa')

df_versicolor = iris.copy()
df_versicolor['species'] = df_versicolor['species'].where(df_versicolor['species'] == 'Iris-versicolor', 'No-versicolor')

df_virginica = iris.copy()
df_virginica['species'] = df_virginica['species'].where(df_virginica['species'] == 'Iris-virginica', 'No-virginica')




# Seleccionar las columnas de interés
X = df_setosa[['sepal_length', 'sepal_width']]
y = df_setosa['species']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("---------------------------------------------------SETOSA--------------------------------------------------------")
# Mostrar los conjuntos de entrenamiento y prueba
print("Conjunto de entrenamiento:")
print(X_train)
print(y_train)

print("\nConjunto de prueba:")
print(X_test)
print(y_test)

# Suponiendo que ya tienes X_train e y_train definidos
setosa_indices = y_train[y_train == 'Iris-setosa'].index

# Filtrar las filas correspondientes a 'setosa' en X_train
setosa_sepal_length_sum = X_train.loc[setosa_indices, 'sepal_length'].sum()

# Contar el número de instancias de 'setosa' en y_train
num_setosa_instances = len(setosa_indices)

# Calcular el promedio
setosa_sepal_length_average = setosa_sepal_length_sum / num_setosa_instances

# Filtrar las filas correspondientes a 'No-Iris-setosa' en X_train
no_setosa_indices = y_train[y_train != 'Iris-setosa'].index

# Filtrar las filas correspondientes a 'No-Iris-setosa' en X_train
no_setosa_sepal_length_sum = X_train.loc[no_setosa_indices, 'sepal_length'].sum()

# Contar el número de instancias de 'No-Iris-setosa' en y_train
num_no_setosa_instances = len(no_setosa_indices)

# Calcular el promedio
no_setosa_sepal_length_average = no_setosa_sepal_length_sum / num_no_setosa_instances

# Filtrar las filas correspondientes a 'setosa' en X_train
setosa_sepal_width_sum = X_train.loc[setosa_indices, 'sepal_width'].sum()

# Calcular el promedio
setosa_sepal_width_average = setosa_sepal_width_sum / num_setosa_instances

# Filtrar las filas correspondientes a 'No-Iris-setosa' en X_train
no_setosa_sepal_width_sum = X_train.loc[no_setosa_indices, 'sepal_width'].sum()

# Calcular el promedio
no_setosa_sepal_width_average = no_setosa_sepal_width_sum / num_no_setosa_instances

v_positivo = [setosa_sepal_length_average,setosa_sepal_width_average]
v_negativo = [no_setosa_sepal_length_average,no_setosa_sepal_width_average]

print(f"\nLos vectores de soporte son: \n{v_positivo}\n{v_negativo}]")

v_perpendicularx = (v_positivo[0] + v_negativo[0]) / 2
v_perpendiculary = (v_positivo[1] + v_negativo[1]) / 2

print(f"\nVector perpendicular: [{v_perpendicularx}, {v_perpendiculary}] ")


# Definir el vector perpendicular (como se sugirió anteriormente)
v_perpendicular = [v_perpendicularx, v_perpendiculary]

projections_setosa = np.array([np.dot(point, v_perpendicular) / np.linalg.norm(v_perpendicular) for point in X_test[['sepal_length', 'sepal_width']].values])

# Establecer un umbral (puedes ajustarlo según tus necesidades)
umbral = sqrt(pow(v_perpendicular[0],2)+pow(v_perpendicular[1],2))

print(f"Magnitud de C setosa: {umbral}")

# Clasificar las proyecciones
predicciones = np.where(projections_setosa > umbral, 'Iris-setosa', 'No-setosa')


#---------------------------------------------------------------------VERSICOLOR----------------------------------------------------------------------------------


# Seleccionar las columnas de interés
X = df_versicolor[['sepal_length', 'sepal_width']]
y = df_versicolor['species']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (80-20)
Xv_train, Xv_test, Yv_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("----------------------------------------------------------VERSICOLOR---------------------------------------------------------------------------------")
# Mostrar los conjuntos de entrenamiento y prueba
print("Conjunto de entrenamiento:")
print(Xv_train)
print(Yv_train)

print("\nConjunto de prueba:")
print(Xv_test)
print(y_test)

# Suponiendo que ya tienes Xv_train e Yv_train definidos
versicolor_indices = Yv_train[Yv_train == 'Iris-versicolor'].index

# Filtrar las filas correspondientes a 'versicolor' en Xv_train
versicolor_sepal_length_sum = Xv_train.loc[versicolor_indices, 'sepal_length'].sum()

# Contar el número de instancias de 'versicolor' en Yv_train
num_versicolor_instances = len(versicolor_indices)

# Calcular el promedio
versicolor_sepal_length_average = versicolor_sepal_length_sum / num_versicolor_instances

# Filtrar las filas correspondientes a 'No-Iris-versicolor' en Xv_train
no_versicolor_indices = Yv_train[Yv_train != 'Iris-versicolor'].index

# Filtrar las filas correspondientes a 'No-Iris-versicolor' en Xv_train
no_versicolor_sepal_length_sum = Xv_train.loc[no_versicolor_indices, 'sepal_length'].sum()

# Contar el número de instancias de 'No-Iris-versicolor' en Yv_train
num_no_versicolor_instances = len(no_versicolor_indices)

# Calcular el promedio
no_versicolor_sepal_length_average = no_versicolor_sepal_length_sum / num_no_versicolor_instances

# Filtrar las filas correspondientes a 'versicolor' en Xv_train
versicolor_sepal_width_sum = Xv_train.loc[versicolor_indices, 'sepal_width'].sum()

# Calcular el promedio
versicolor_sepal_width_average = versicolor_sepal_width_sum / num_versicolor_instances

# Filtrar las filas correspondientes a 'No-Iris-versicolor' en Xv_train
no_versicolor_sepal_width_sum = Xv_train.loc[no_versicolor_indices, 'sepal_width'].sum()

# Calcular el promedio
no_versicolor_sepal_width_average = no_versicolor_sepal_width_sum / num_no_versicolor_instances


v_positivo = [versicolor_sepal_length_average,versicolor_sepal_width_average]
v_negativo = [no_versicolor_sepal_length_average,no_versicolor_sepal_width_average]

print(f"\nLos vectores de soporte son: \n{v_positivo}\n{v_negativo}]")


v_perpendicularx = (v_positivo[0] + v_negativo[0]) / 2
v_perpendiculary = (v_positivo[1] + v_negativo[1]) / 2

print(f"\nVector perpendicular: [{v_perpendicularx}, {v_perpendiculary}] ")


# Definir el vector perpendicular (como se sugirió anteriormente)
v_perpendicular = [v_perpendicularx, v_perpendiculary]

projections_versicolor = np.array([np.dot(point, v_perpendicular) / np.linalg.norm(v_perpendicular) for point in Xv_test[['sepal_length', 'sepal_width']].values])

# Establecer un umbral_versicolor (puedes ajustarlo según tus necesidades)
umbral_versicolor = sqrt(pow(v_perpendicular[0],2)+pow(v_perpendicular[1],2))

print(f"Magnitud de C Versicolor: {umbral_versicolor}")

# Clasificar las proyecciones
predicciones_versicolor = np.where(projections_versicolor > umbral_versicolor, 'Iris-versicolor', 'No-versicolor')



#-----------------------------------------------------------VIRGINICA------------------------------------------------------------------------------

# Seleccionar las columnas de interés
X = df_virginica[['sepal_length', 'sepal_width']]
y = df_virginica['species']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (80-20)
Xvir_train, Xvir_test, Yvir_train, Yvir_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("------------------------------------------------------VIRGINICA--------------------------------------------------------------------")
# Mostrar los conjuntos de entrenamiento y prueba
print("Conjunto de entrenamiento:")
print(Xvir_train)
print(Yvir_train)

print("\nConjunto de prueba:")
print(Xvir_test)
print(Yvir_test)

# Suponiendo que ya tienes Xvir_train e Yvir_train definidos
virginica_indices = Yvir_train[Yvir_train == 'Iris-virginica'].index

# Filtrar las filas correspondientes a 'virginica' en Xvir_train
virginica_sepal_length_sum = Xvir_train.loc[virginica_indices, 'sepal_length'].sum()

# Contar el número de instancias de 'virginica' en Yvir_train
num_virginica_instances = len(virginica_indices)

# Calcular el promedio
virginica_sepal_length_average = virginica_sepal_length_sum / num_virginica_instances

# Filtrar las filas correspondientes a 'No-Iris-virginica' en Xvir_train
no_virginica_indices = Yvir_train[Yvir_train != 'Iris-virginica'].index

# Filtrar las filas correspondientes a 'No-Iris-virginica' en Xvir_train
no_virginica_sepal_length_sum = Xvir_train.loc[no_virginica_indices, 'sepal_length'].sum()

# Contar el número de instancias de 'No-Iris-virginica' en Yvir_train
num_no_virginica_instances = len(no_virginica_indices)

# Calcular el promedio
no_virginica_sepal_length_average = no_virginica_sepal_length_sum / num_no_virginica_instances

# Filtrar las filas correspondientes a 'virginica' en Xvir_train
virginica_sepal_width_sum = Xvir_train.loc[virginica_indices, 'sepal_width'].sum()

# Calcular el promedio
virginica_sepal_width_average = virginica_sepal_width_sum / num_virginica_instances

# Filtrar las filas correspondientes a 'No-Iris-virginica' en Xvir_train
no_virginica_sepal_width_sum = Xvir_train.loc[no_virginica_indices, 'sepal_width'].sum()

# Calcular el promedio
no_virginica_sepal_width_average = no_virginica_sepal_width_sum / num_no_virginica_instances


v_positivo = [virginica_sepal_length_average,virginica_sepal_width_average]
v_negativo = [no_virginica_sepal_length_average,no_virginica_sepal_width_average]

print(f"\nLos vectores de soporte son: \n{v_positivo}\n{v_negativo}]")


v_perpendicularx = (v_positivo[0] + v_negativo[0]) / 2
v_perpendiculary = (v_positivo[1] + v_negativo[1]) / 2

print(f"\nVector perpendicular de Virginica: [{v_perpendicularx}, {v_perpendiculary}] ")


# Definir el vector perpendicular (como se sugirió anteriormente)
v_perpendicular = [v_perpendicularx, v_perpendiculary]

projections_virginica = np.array([np.dot(point, v_perpendicular) / np.linalg.norm(v_perpendicular) for point in Xvir_test[['sepal_length', 'sepal_width']].values])

# Establecer un umbral_virginica (puedes ajustarlo según tus necesidades)
umbral_virginica = sqrt(pow(v_perpendicular[0],2)+pow(v_perpendicular[1],2))
print(f"Magnitud de C de Virgnica: {umbral_virginica}")

predicciones_virginica = np.where(projections_virginica > umbral_virginica, 'Iris-virginica', 'No-virginica')

#------------------------------------------PROYECCIONES DE 3 CLASES--------------------------------------------------------------------

# Crear un DataFrame con las proyecciones
proyecciones_df = pd.DataFrame({
    'Setosa': projections_setosa,
    'Versicolor': projections_versicolor,
    'Virginica': projections_virginica,
})

print("\n\n\n----------------------------------------------------Conjuntos de Prueba y Prueba Final IRIS----------------------------------------------")

# Mostrar el DataFrame
print("\nProyecciones del Conjunto de Prueba:")
print(proyecciones_df)



sign_setosa = np.sign(projections_setosa - umbral)
sign_versicolor = np.sign(projections_versicolor - umbral_versicolor)
sign_virginica = np.sign(projections_virginica - umbral_virginica)

# Determine if magnitudes are greater or smaller than the perpendicular vector
mayor_setosa = "No-setosa" if np.linalg.norm(v_positivo) >= np.linalg.norm(v_perpendicular) else "Iris-setosa"
mayor_versicolor = "No-versicolor" if np.linalg.norm(v_positivo) >= np.linalg.norm(v_perpendicular) else "Iris-versicolor"
mayor_virginica = "No-virginica" if np.linalg.norm(v_positivo) >= np.linalg.norm(v_perpendicular) else "Iris-virginica"

# Make predictions based on signs and magnitudes
pred_setosa = np.where(sign_setosa == 1, mayor_setosa, 'Iris-setosa')
pred_versicolor = np.where(sign_versicolor == 1, mayor_versicolor, 'Iris-versicolor')
pred_virginica = np.where(sign_virginica == 1, 'Iris-virginica', mayor_virginica)

# Display predictions vertically
print("\nPredicciones de cada conjunto de Prueba:")
# Display predictions in a table
table = PrettyTable()
table.field_names = ["Setosa", "Versicolor", "Virginica"]

for p_setosa, p_versicolor, p_virginica in zip(pred_setosa, pred_versicolor, pred_virginica):
    table.add_row([p_setosa, p_versicolor, p_virginica])

print(table)



# Calculate probabilities for each class
prob_setosa = num_setosa_instances / 120
prob_versicolor = num_versicolor_instances / 120
prob_virginica = num_virginica_instances / 120

print(f"\nPobabilidad de Setosa: ", prob_setosa)
print(f"Pobabilidad de Versicolor: ", prob_versicolor)
print(f"Pobabilidad de Virginica: ", pred_virginica)

# Create a DataFrame with the probabilities
probs_df = pd.DataFrame({
    'Setosa': prob_setosa,
    'Versicolor': prob_versicolor,
    'Virginica': prob_virginica,
}, index=X_test.index)  


# Mapear las predicciones finales a las clases deseadas
final_predictions = []

for setosa, versicolor, virginica in zip(pred_setosa, pred_versicolor, pred_virginica):
    if setosa == 'Iris-setosa' and versicolor == 'Iris-versicolor' and virginica == 'Iris-virginica':
        final_predictions.append('Iris-setosa')
    elif setosa == 'No-setosa' and versicolor == 'No-versicolor' and virginica == 'No-virginica':
        final_predictions.append('Iris-setosa')
    elif setosa == 'No-setosa' and versicolor == 'Iris-versicolor' and virginica == 'Iris-virginica':
        final_predictions.append('Iris-virginica')
    elif setosa == 'No-setosa' and versicolor == 'No-versicolor' and virginica == 'Iris-virginica':
        final_predictions.append('Iris-virginica')
    elif setosa == 'Iris-setosa' and versicolor == 'No-versicolor' and virginica == 'No-virginica':
        final_predictions.append('Iris-setosa')
    elif setosa == 'No-setosa' and versicolor == 'Iris-versicolor' and virginica == 'No-virginica':
        final_predictions.append('Iris-versicolor')
    elif setosa == 'Iris-setosa' and versicolor == 'Iris-versicolor' and virginica == 'No-virginica':
        final_predictions.append('Iris-setosa')
    else:
        final_predictions.append('Desconocido')

# Display final predictions in a table
final_table = PrettyTable()
final_table.field_names = ["Predicciones Finales"]

for prediction in final_predictions:
    final_table.add_row([prediction])

print("\nPredicciones Finales:")
print(final_table)


irisc_data = pd.read_csv('iris.csv')
df = pd.DataFrame(irisc_data)

# Seleccionar las columnas de interés
X = df[['sepal_length', 'sepal_width']]
y = df['species']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (80-20)
Xi_train, Xi_test, yi_train, yi_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("\n\nPRUEBA IRIS")
print(yi_test)

print("\nReporte de Clasificación Final:")
print(classification_report(yi_test, final_predictions))

conf_matrix = confusion_matrix(yi_test, final_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Setosa', 'Versicolor', 'Virginica'],  
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])  
plt.title('Matriz de Confusión Final')
plt.xlabel('Predicciones')
plt.ylabel('Valores Verdaderos')
plt.show()


print("Tabla")

# Create a PrettyTable instance
table = PrettyTable()

# Define table columns
table.field_names = ["SVM Configuration", "Accuracy"]

table.add_row(["SVM con vectores de soporte medio", .40])
table.add_row(["SVM con Kernel Lineal", .80])
table.add_row(["SVM de Base Radial", .8266666666666667])
table.add_row(["SVM polinomial", .8133333333333334])

# Print the table
print(table)