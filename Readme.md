<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps) - Mauro Ferrera'**</h1>

<p align="center">
<img src="https://tse3.mm.bing.net/th?id=OIP.aSC3odScNMyz7Y6MZvqJ1QHaEK&pid=Api&P=0&h=180"  height=300>
</p>

## ¡Bienvenido/a a este proyecto donde te explicare como hacer un proceso de ETL, un analisis EDA y un modelo de ML(Machine learning) para hacer un sistema de recomendacion trabajando con un datasets de peliculas
 

<hr>  

## **Descripción del problema (Contexto y rol a desarrollar)**

## Contexto
### Tenemos 1 datasets que contiene la siguiente informacion:  
#### adult:	                Indica si la película tiene califiación X, exclusiva para adultos.			
#### belongs_to_collection:	Un diccionario que indica a que franquicia o serie de películas pertenece la película			
#### budget:	            El presupuesto de la película, en dólares			
#### genres:	            Un diccionario que indica todos los géneros asociados a la película			
#### homepage:	            La página web oficial de la película			
#### id:	                ID de la pelicula			
#### imdb_id:	            IMDB ID de la pelicula			
#### original_language:	    Idioma original en la que se grabo la pelicula			
#### original_title:	    Titulo original de la pelicula			
#### overview:	            Pequeño resumen de la película			
#### popularity:	        Puntaje de popularidad de la película, asignado por TMDB (TheMoviesDataBase)			
#### poster_path:	        URL del póster de la película			
#### production_companies:	Lista con las compañias productoras asociadas a la película			
#### production_countries:	Lista con los países donde se produjo la película			
#### release_date:	        Fecha de estreno de la película			
#### revenue:	            Recaudación de la pelicula, en dolares			
#### runtime:	            Duración de la película, en minutos			
#### spoken_languages:	    Lista con los idiomas que se hablan en la pelicula			
#### status:	            Estado de la pelicula actual (si fue anunciada, si ya se estreno, etc)			
#### tagline:	            Frase celebre asociadaa la pelicula			
#### title:	                Titulo de la pelicula			
#### video:	                Indica si hay o no un trailer en video disponible en TMDB			
#### vote_average:	        Puntaje promedio de reseñas de la pelicula			
#### vote_count:	        Numeros de votos recibidos por la pelicula, en TMDB			 

<hr>  

# El rol que cumpliremos en este proyecto es de un DataSciense 

### Paso 1. **`Transformaciones de los datos(ETL)`** : 


+ Algunos campos, como belongs_to_collection, production_companiesy otros (ver diccionario de datos) están anidados, esto es o bien tienen un diccionario o una lista como valores en cada fila, ¡deberán desinfectarlos para poder y unirlos al dataset de nuevo hacer alguna de las consultas de la API! O bien buscar la manera de acceder a esos datos sin desinfectarlos.

Los valores nulos de los campos revenue, budgetdeben ser rellenados por el número 0.

Los valores nulos del campo release datedeben eliminarse.

De haber fechas, deberá tener el formato AAAA-mm-dd, además deberá crear la columna release_yeardonde extraerán el año de la fecha de estreno.

Cree la columna con el retorno de inversión, llame returncon los campos revenuey budget, dividiendo estas dos últimas revenue / budget, cuando no haya datos disponibles para calcularlo, deberá tomar el valor 0.

Eliminar las columnas que no se utilizarán, video, imdb_id, adult, original_title, vote_count, poster_pathy homepage
 
                 **  El codigo que puede encontrar en el archivo: 'Proceso_ETL.ipynb' **
<br/>

<hr>

### Paso 2. **`Desarrollamos una API`** en donde vamos a poder hacer consultas de las siguientes tipo
 ***Disponibilizamos los datos usando el framework*** ***FastAPI***.
   ***Las consultas son las siguientes:***


#### def peliculas_mes(mes): '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes (nombre del mes, en str, ejemplo 'enero') historicamente''' return {'mes':mes , 'cantidad':respuesta}

#### def peliculas_dia(dia): '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia (de la semana, en str, ejemplo 'lunes') historicamente''' return {'dia':dia , 'cantidad':respuesta}

#### def franquicia(franquicia): '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio''' return {'franquicia':franquicia, 'cantidad':respuesta, 'ganancia_total':respuesta, 'ganancia_promedio' :respuesta}

#### def peliculas_pais(pais): '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo''' return {'pais':pais, 'cantidad':respuesta}

#### def productoras(productora): '''Ingresas la productora, retornando la ganancia total y la cantidad de peliculas que producen'' return {'productora':productora, 'ganancia_total':respuesta, 'cantidad':respuesta}

#### def retorno(pelicula): '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo'' return {'pelicula':pelicula, 'inversion':respuesta, 'ganacia' :respuesta,'retorno':respuesta, 'año':respuesta}

                   ** Al codigo de las consultas se puede encontrar en el archivo main.py ** 

<br/>

<hr>




### Paso 3. **`Análisis exploratorio de los datos`** _(Exploratory Data Analysis-EDA)_:



😉 Ya los datos están limpios, ahora es tiempo de investigar las relaciones que hay entre las variables de los datasets, ver si hay outliers o anomalías (que no tienen que ser errores necesariamente :eyes: ), y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior.  Nos podemos apoyar en librerías como: _pandas profiling, sweetviz, autoviz_, entre otros y sacar nuestras propias conclusiones 
                  
                   ** El codigo explicado paso a paso del analisis EDA se encuentra en el archivo 'ETL_EDA_PI_MLops.ipynb' **

<hr>

### Paso  4. Creamos el **`Sistema de recomendacion`** con un modelo de Machine learning: 

Una vez que toda la data es consumible por la API y nuestro EDA bien realizado entendiendo bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un sistema de recomendación de películas para usuarios. Éste consiste en recomendar películas a los usuarios en películas similares, por lo que se debe encontrar la similitud de puntuación entre esa película y el resto de películas, se ordenarán la partitura y devolverá según la lista de Python con 5 valores .luego lo implementamos implementado como una función adicional de la API anterior y debe llamarse 'get_recommendation(titulo: str)'. De ser posible, este sistema de recomendación debe ser deployado para tener una interfaz gráfica amigable para ser utilizada
**Para hacer este sistema debemos vectorizar una columna texto que va a hacer las comparaciones de palabras o caracteristicas entre las peliculas, descomponer la matriz si el archivo pesa demasiado, obtener la similitud del coseno y luego crear la funcion 'get_recommendation(titulo: str)' con la matriz que nos da la similitud del coseno**  
<br/>
                    ** Al codigo explicado paso a paso lo encontramos en el archivo, 'main.py' ** 
<hr>

### **Video explicativo de todos los pasos**



<br/>

<hr>

### **Deployemend:** 


