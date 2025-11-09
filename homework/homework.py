#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_kms: Kilometraje recorrido.
# - Fuel_Type: Tipo de combustible.
# - Selling_type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import numpy as np
import gzip
import pickle
import os

test_data = pd.read_csv(
        f"files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )
train_data = pd.read_csv(
        f"files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )
## Paso 1 Limpiar la data

def clean_data(df):
    #1. Calcular la columna Age
    df['Age']=2021-df['Year']
    # 2. Eliminar la columna 'Year' y 'Car_name' si existe
    if 'Car_Name' in df.columns:
        df = df.drop(columns=['Car_Name'])
    if 'Year' in df.columns:
        df = df.drop(columns=['Year'])
    # 3. Eliminar filas con valores NaN
    df = df.dropna()
    #4. Convertir a log el precio
    df['Selling_Price']=np.log1p(df['Selling_Price'])
    
    return df

train_data=clean_data(train_data)
test_data=clean_data(test_data)

## Paso 2 Dividir la data
x_train = train_data.drop(columns=['Present_Price'])
y_train = np.log1p(train_data['Present_Price'])

x_test = test_data.drop(columns=['Present_Price'])
y_test = np.log1p(test_data['Present_Price'])


## Paso 3 Creación del pipepline y ajuste de modelo
def make_pipeline(estimator):

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile

# - Fuel_Type: Tipo de combustible.
# - Selling_type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
    cols=['Fuel_Type','Selling_type','Transmission']
    cols_num = ["Selling_Price", "Driven_kms", "Owner", "Age"] 
    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"),cols),
            ("num", MinMaxScaler(), cols_num),
        ],
        remainder="drop",
    )
    selectkbest=SelectKBest(score_func=f_regression)
    
    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False
    )

    return pipeline

# Modelo estimador
from sklearn.linear_model import LinearRegression
rgr=LinearRegression()
rgr_estimator=make_pipeline(rgr)

## Paso 4 Optimizar hiperparámetros 10 splits
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error
import numpy as np

# Grid mínimo para cumplir la especificación
param_grid = {
    'selectkbest__k': [9,10,11],
    'estimator__fit_intercept':[True,False]
}

cv = KFold(n_splits=10, shuffle=False)
#scorers = {
    #"acc": "accuracy",
    #"abs": mean_absolute_error
#}

grid_search = GridSearchCV(estimator=rgr_estimator,                       
    param_grid=param_grid,
    scoring='neg_mean_absolute_error', 
    refit=True,
    cv=cv,
    n_jobs=-1,
    verbose=0,
    error_score="raise"
    )


## Paso 5 Guardar el modelo comprimido
grid_search.fit(x_train,y_train)

print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

print("Mejor puntaje CV (scoring):")
print(grid_search.best_score_)


def save_estimator(best_estimator):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(best_estimator, f)

save_estimator(grid_search)

best_est = grid_search.best_estimator_
print(best_est)

## Paso 6 Métricas de precisión
import json
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

with gzip.open("files/models/model.pkl.gz", "rb") as f:
    loaded_model = pickle.load(f)

y_train_pred = np.expm1(grid_search.predict(x_train))
y_test_pred  = np.expm1(grid_search.predict(x_test))


y_train_true = np.expm1(y_train)
y_test_true  = np.expm1(y_test)

metrics = [
    {"type":"metrics","dataset":"train",
     "r2": r2_score(y_train_true, y_train_pred),
     "mse": mean_squared_error(y_train_true, y_train_pred),
     "mad": mean_absolute_error(y_train_true, y_train_pred)},
    {"type":"metrics","dataset":"test",
     "r2": r2_score(y_test_true, y_test_pred),
     "mse": mean_squared_error(y_test_true, y_test_pred),
     "mad": mean_absolute_error(y_test_true, y_test_pred)}
]

os.makedirs("files/output", exist_ok=True)

with open("files/output/metrics.json", "w") as f:
    for row in metrics:
        f.write(json.dumps(row) + "\n")

