#Rodrigo Palomo, Inés Salcedo y Juan Velayos.

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Carga el archivo Excel
file_path = 'Accidentalidad.xlsx'
data = pd.read_excel(file_path)

# Crear un subconjunto con las columnas relevantes
subset = data[['num_expediente', 'sexo', 'tipo_accidente', 'lesividad']]

#limpiamos los NAs para no tener problemas en el analisis.
subset = subset.dropna()

# Calcular la tabla de correlación
correlation_table = data.corr()

# Imprimir la tabla de correlación
print(correlation_table)

# Crear una representación gráfica de la correlación
sns.heatmap(correlation_table, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#ESTUDIO ESTADISTICO
## Estudio estadístico del número de accidentes en función del sexo
accidents_by_gender = subset.groupby('sexo')['num_expediente'].count()
print(accidents_by_gender)

# Crear una gráfica de barras para representar el estudio estadístico del número de accidentes en función del sexo
accidents_by_gender.plot(kind='bar')
plt.title('Accidents by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Accidents')
plt.show()

## Estudio estadístico del tipo de accidente
accidents_by_type = subset.groupby('tipo_accidente')['num_expediente'].count()
print(accidents_by_type)

# Crear una gráfica de barras para representar el estudio estadístico del tipo de accidente
accidents_by_type.plot(kind='bar')
plt.title('Accidents by Type')
plt.xlabel('Accident Type')
plt.ylabel('Number of Accidents')
plt.show()

## Estudio estadístico de la lesividad
accidents_by_lesion = subset.groupby('lesividad')['num_expediente'].count()
print(accidents_by_lesion)

# Crear una gráfica de barras para representar el estudio estadístico de la lesividad
accidents_by_lesion.plot(kind='bar')
plt.title('Accidents by Lesion')
plt.xlabel('Lesion')
plt.ylabel('Number of Accidents')
plt.show()

## Estudio estadístico del tipo de accidente
accidents_by_type = subset.groupby('tipo_accidente')['num_expediente'].count()
print(accidents_by_type)

# Crear una gráfica de barras para representar el estudio estadístico del tipo de accidente
accidents_by_type.plot(kind='bar')
plt.title('Accidents by Type')
plt.xlabel('Accident Type')
plt.ylabel('Number of Accidents')
plt.show()

## Estudio estadístico de la lesividad
accidents_by_lesion = subset.groupby('lesividad')['num_expediente'].count()
print(accidents_by_lesion)

# Crear una gráfica de barras para representar el estudio estadístico de la lesividad
accidents_by_lesion.plot(kind='bar')
plt.title('Accidents by Lesion')
plt.xlabel('Lesion')
plt.ylabel('Number of Accidents')
plt.show()




# Modelado Estadístico

# Modelos de regresión
import statsmodels.api as sm

# Crear variables predictoras y variable objetivo
X = subset[['sexo', 'tipo_accidente']]
y = subset['lesividad']

# Agregar una columna de unos para el intercepto
X = sm.add_constant(X)

# Ajustar el modelo de regresión
model = sm.OLS(y, X).fit()

# Imprimir los resultados del modelo
print(model.summary())

# Modelos de clasificación

# Crear variables predictoras y variable objetivo
X = subset[['sexo', 'tipo_accidente']]
y = subset['lesividad']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y ajustar el modelo de clasificación
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir las etiquetas de clase para los datos de prueba
y_pred = model.predict(X_test)

# Imprimir el informe de clasificación
print(classification_report(y_test, y_pred))