#Sergio Terrazas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Cargar el conjunto de datos
data = pd.read_csv("datos_divorcio.csv")

# Mapear las respuestas a los valores correspondientes
mapeo_respuestas = {
    "En desacuerdo totalmente": 1,
    "En desacuerdo": 2,
    "Irrelevante": 3,
    "De acuerdo": 4,
    "De acuerdo totalmente": 5
}

data.replace(mapeo_respuestas, inplace=True)

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear una instancia del modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)

# Calcular la matriz de confusión
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion_matrix)

# Generar un informe de clasificación
classification_report = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(classification_report)
