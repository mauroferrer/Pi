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

# Hacemos una introduccion donde va a aparecer nuestro nombre en este caso el mio, Mauro Ferrera 
@app.get("/")
def presentacion():
    return {"PI_MLops - Mauro Ferrera"}


####CONSULTA 3
# Esta funcion nos va a retornar la cantidad de peliculas que tiene la franquicia que le indiquemos, su ganancia total y ganancia promedio 
@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    franquicias_df = data.loc[data['name_production'] == franquicia]#asignamos los datos de la columna 'name_production' a el parametro franquicia
    cantidad = franquicias_df.shape[0] #contamos la cantidad de registros que aparece la franquicia en la columna 'nameproduction'
    ganancia_total = franquicias_df['revenue'].sum()#calculamos la suma de la columna 'reveue'
    ganancia_promedio = ganancia_total / cantidad #sacamos el promedio de las ganancias
    return{'franquicia':franquicia, 'cantidad':cantidad, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}



 ####CONSULTA 4
 #Esta funcion nos retorna la cantidad de peliculas que se crearon en el pais que le pasemos como parametro
@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    pais_data = data.loc[data['name_countrie'] == pais]#asignamos la columna 'name_countrie' al parametro 'pais'
    cantidad = pais_data['name_countrie'].value_counts()[pais]#Contamos la cantidad de veces que aparece el pais
    return 'pais:',pais ,'cantidad de peliculas producidas',cantidad.item()#retornamos la cantidad de peliculas de ese pais



####CONSULTA 5
#Esta funcion nos retorna la ganancia total y la cantidad de peliculas de la productora que le pasemos como parametro
@app.get('/productoras/{productora}')
def productoras(productora:str):
    df_productora =data.loc[data['name_production'] == productora]#Asignamos la productora a la columna 'name_production'
    ganancia_total = df_productora['revenue'].sum()#El total de ingresos que genero la productora
    peliculas_producidas = df_productora.index.nunique()
    return{'producotora':productora, 'ganancia':ganancia_total, 'cantidad':peliculas_producidas}



####CONSULTA NUMERO 1 
#Creamos una consulta donde segun el mes que indiquemos nos va a arrojar la cantidad de peliculas creadas en ese mes
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



####CONSULTA 6
#Esta funcion retornala inversion dedicada, las ganancias, el retorno y el año de la pelicula que le indiquemos como parametro
@app.get('/retorno/{pelicula}')
def retorno(pelicula:str):
    pelicula_df = data.loc[data['title']== pelicula.title()]#Asignamos titulo al parametro pelicula
    inversion = pelicula_df['budget'].iloc[0].item()#Obtenemos la inversion de la pelicula
    ganancia = pelicula_df['revenue'].iloc[0].item()#Obtenemos las ganancias de la pelicula
    retorno = pelicula_df['return'].iloc[0].item()#Obtenemos el retorno de la pelicula
    anio = pelicula_df['release_year'].iloc[0].item()#Obtenemos el año de la pelicula
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
    