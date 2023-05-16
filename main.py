  ####Importamos librerias 
import locale
from fastapi import FastAPI, HTTPException #Para crear la api
import pandas as pd # Manejo de dataframes 
from typing import Optional
import uvicorn  # Para correr nuestra API
from sklearn.metrics.pairwise import cosine_similarity #Utilizamos para obtener la similitud del coseno 
from sklearn.utils.extmath import randomized_svd # Utilizamos SVD para desponer nuestra matriz 
from sklearn.feature_extraction.text import  TfidfVectorizer #Utilizamos para vectorizar datos tipo texto y convertirlos en una representacion numerica
import numpy as np # Manejo de matrices, array, etc

app = FastAPI()


data = pd.read_csv('df_arreglado1.csv')

# introduccion
@app.get("/")
def presentacion():
    return {"PI_MLops - Mauro Ferrera"}

@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes):
    fechas = pd.to_datetime(data['release_date'], format= '%Y-%m-%d')
    n_mes= fechas[fechas.dt.month_name(locale = 'es')==mes.capitalize()]
    respuesta = n_mes.shape[0]
    return {'mes':mes, 'cantidad':respuesta}

@app.get('/peliculas_dis/{dis}')
def peliculas_dia(dia):
    fechas = pd.to_datetime(data['release_date'], format= '%Y-%m-%d')
    n_dia= fechas[fechas.dt.day_name(locale = 'es')==dia.capitalize()]
    respuesta = n_dia.shape[0]
    return {'dia':dia, 'cantidad':respuesta}

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    franquicias_df = data.loc[data['name_production'] == franquicia]
    cantidad = franquicias_df.shape[0]
    ganancia_total = franquicias_df['revenue'].sum()
    ganancia_promedio = ganancia_total / cantidad
    return{'franquicia':franquicia, 'cantidad':cantidad, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}
 
@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    pais_data = data.loc[data['name_countrie'] == pais]
    cantidad = pais_data['name_countrie'].value_counts()[pais]
    return 'pais:',pais ,'cantidad de peliculas producidas',cantidad.item()


@app.get('/productoras/{productora}')
def productoras(productora:str):
    df_productora =data.loc[data['name_production'] == productora]
    ganancia_total = df_productora['revenue'].sum()
    peliculas_producidas = df_productora.index.nunique()
    return{'producotora':productora, 'ganancia':ganancia_total, 'cantidad':peliculas_producidas}

@app.get('/retorno/{pelicula}')
def retorno(pelicula:str):
    pelicula_df = data.loc[data['title']== pelicula.title()]
    inversion = pelicula_df['budget'].iloc[0].item()
    ganancia = pelicula_df['revenue'].iloc[0].item()
    retorno = pelicula_df['return'].iloc[0].item()
    anio = pelicula_df['release_year'].iloc[0].item()
    return{
        'pelicula':pelicula,
        'inversion':inversion,
        'ganancia':ganancia,
        'retorno':retorno,
        'anio':anio
    }


#### Creamos la matriz de similitud del coseno ####

# Vectorizador TfidfVectorizer con parámetros de reduccion procesamiento
data['name_genres'].fillna('', inplace=True)
vectorizer = TfidfVectorizer(min_df=10, max_df=0.5, ngram_range=(1,2))

# Vectorizar, ajustar y transformar el texto de la columna "title" del DataFrame
X = vectorizer.fit_transform(data['name_genres'])

# Calcular la matriz de similitud de coseno con una matriz reducida de 7000
similarity_matrix = cosine_similarity(X[:1250,:])

# Obtener la descomposición en valores singulares aleatoria de la matriz de similitud de coseno con 10 componentes
n_components = 10
U, Sigma, VT = randomized_svd(similarity_matrix, n_components=n_components)

# Construir la matriz reducida de similitud de coseno
reduced_similarity_matrix = U.dot(np.diag(Sigma)).dot(VT)

#ML
@app.get('/recomendacion/{titulo}')
def get_recommendation(titulo: str):
    try:
        #Ubicamos el indice del titulo pasado como parametro en la columna 'title' del dts user_item
        indice = np.where(data['title'] == titulo)[0][0]
        #Encontramos los indices de las puntuaciones y caracteristicas similares del titulo 
        puntuaciones_similitud = reduced_similarity_matrix[indice,:]
        #Ordenamos los indices de menor a mayor
        puntuacion_ordenada = np.argsort(puntuaciones_similitud)[::-1]
        #seleccionamos solo 5 
        top_indices = puntuacion_ordenada[:5]
        #retornamos los 5 items con sus titulos como una lista
        return data.loc[top_indices, 'title'].tolist()
        #Si el titulo dado no se encuentra damos un aviso
    except IndexError:
        print(f"El título '{titulo}' no se encuentra en la base de datos. Intente con otro título.")
 
 
if __name__ == "_main_":

    uvicorn.run(app, host="0.0.0.0", port=1000)
    