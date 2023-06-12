from typing import Union
from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from typing import Dict, Any

app = FastAPI(title='Proyecto I')

# Carga los datos una sola vez al iniciar la aplicación
dfETL = pd.read_csv('movies_ETL.csv', parse_dates=['release_date'])
#dfr1 = dfETL[['id', 'title', 'genres', 'overview']].fillna({'overview': ' '})

# Crea el vectorizador y ajusta el texto
#cv = CountVectorizer(max_features=6500, stop_words='english')
#vectors = cv.fit_transform(dfr1['overview']).toarray()

#ps = PorterStemmer()
#similarity = cosine_similarity(vectors)

@app.get("/cantidad_filmaciones_mes/")
def cantidad_filmaciones_mes(mes: str):
    data = dfETL

    meses = {
        'enero': 1,
        'febrero': 2,
        'marzo': 3,
        'abril': 4,
        'mayo': 5,
        'junio': 6,
        'julio': 7,
        'agosto': 8,
        'septiembre': 9,
        'octubre': 10,
        'noviembre': 11,
        'diciembre': 12
    }

    num_mes = meses.get(mes.lower())

    if num_mes is None:
        raise ValueError('Nombre de mes inválido')

    filmaciones_mes = data[data['release_date'].dt.month == num_mes]
    cantidad_filmaciones = len(filmaciones_mes)

    return f" mes: {mes.capitalize()}, cantidad: {cantidad_filmaciones}"


@app.get("/cantidad_filmaciones_dia/")
def cantidad_filmaciones_dia(dia: str):
    data = dfETL

    data['release_date'] = pd.to_datetime(data['release_date'])

    dias = {
        'lunes': 0,
        'martes': 1,
        'miercoles': 2,
        'jueves': 3,
        'viernes': 4,
        'sabado': 5,
        'domingo': 6
    }

    num_dia = dias.get(dia.lower())

    if num_dia is None:
        raise ValueError('Nombre de día inválido')

    filmaciones_dia = data[data['release_date'].dt.weekday == num_dia]
    cantidad_filmaciones = len(filmaciones_dia)

    return {f" dia:{dia.capitalize()}, cantidad: {cantidad_filmaciones} "}

@app.get("/score_titulo/")
def score_titulo(titulo: str):
    data = dfETL

    titulo = titulo.title()

    match = data[data['title'] == titulo]

    if not match.empty:
        titulo1 = match['title'].values[0]
        anio = match['release_year'].values[0]
        score_popularity = match['popularity'].values[0]

        score_popularity = round(score_popularity, 2)

        return {f"titulo: {titulo1}, año: {anio}, popularidad: {score_popularity}"}
        
    return {"mensaje": "No se encontró la película especificada."}

@app.get("/votos_titulo/")
def votos_titulo(titulo: str):
    data = dfETL
    data['title'] = data['title'].str.title()
    titulo = titulo.title()

    if data['title'].str.contains(titulo).any():
        if data[data['title'] == titulo]['vote_count'].item() < 2000:
            return {"mensaje": "La película no cuenta con al menos 2000 valoraciones."}
        else:
            titulo1 = titulo
            anio = data[data['title'] == titulo]['release_year'].item()
            c_votos = data[data['title'] == titulo]['vote_count'].item()
            score = data[data['title'] == titulo]['vote_average'].item()

            c_votos = int(c_votos)


            return { f"titulo: {titulo1}, año: {anio}, voto_total: {c_votos} , voto_promedio de {score}"}
            
    else:
        return {"mensaje": "La película no se encontró."}


@app.get("/get_actor/")
def get_actor(nombre_actor: str):
    data = dfETL
    nombre_a = str(nombre_actor).title()

    if data['name'].apply(lambda x: isinstance(x, list) and nombre_a in x).any():
        c_filmaciones = data[data['name'].apply(lambda x: isinstance(x, list) and nombre_a in x)]['title'].count()
        retorno = data[data['name'].apply(lambda x: isinstance(x, list) and nombre_a in x)]['retorno_inversion'].sum()
        promedio = data[data['name'].apply(lambda x: isinstance(x, list) and nombre_a in x)]['retorno_inversion'].mean()

        retorno = round(retorno, 2)
        promedio = round(promedio, 2)

        return {f" actor: {nombre_a}, cantidad_filmaciones: {c_filmaciones}, retorno_total: {retorno}, voto_promedio: {promedio}"}
        

    elif data['name'].apply(lambda x: isinstance(x, str) and nombre_a in x).any():
        c_filmaciones = data[data['name'].apply(lambda x: isinstance(x, str) and nombre_a in x)]['title'].count()
        retorno = data[data['name'].apply(lambda x: isinstance(x, str) and nombre_a in x)]['retorno_inversion'].sum()
        promedio = data[data['name'].apply(lambda x: isinstance(x, str) and nombre_a in x)]['retorno_inversion'].mean()

        retorno = round(retorno, 2)
        promedio = round(promedio, 2)

        return {f" actor: {nombre_a}, cantidad_filmaciones: {c_filmaciones}, retorno_total: {retorno}, voto_promedio: {promedio}"}
        #return {"mensaje": f"El actor {nombre_a} ha participado en {c_filmaciones} filmaciones, ha conseguido un retorno total de {retorno} con un promedio de {promedio} por filmación."}
    else:
        return {"mensaje": f"El actor {nombre_a} no se encuentra en la base de datos."}




@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str) -> Dict[str, Any]:
    director = dfETL[dfETL['Director'] == nombre_director]

    if director.empty:
        return {"mensaje": "El director no se encuentra en el dataset."}

    exito_director = director.shape[0]

    # Ajustar la columna "release_date" para eliminar la hora
    director['release_date'] = director['release_date'].dt.date

    peliculas_director = director[['title', 'release_date', 'budget', 'revenue']]
    peliculas_director['release_date'] = peliculas_director['release_date'].astype(str)

    return {
        "director": nombre_director,
        "retorno_director": exito_director,
        "peliculas_director": peliculas_director.to_dict(orient='records')
    }


from sklearn.metrics.pairwise import cosine_similarity

# @app.get("/recommend/{movie}")
# def recommend(movie: str):
#     data = dfETL
#     dfr1 = data[['id', 'title', 'genres', 'overview']]

#     # Fill in the missing values in the overview column with a space
#     dfr1 = dfr1.fillna({'overview': ' '})

#     # Create the vectorizer and fit it to the text
#     cv = CountVectorizer(max_features=2000, stop_words='english')

#     # Get the vectors for the overview column
#     vectors = cv.fit_transform(dfr1['overview']).toarray()

#     ps=PorterStemmer()

#     #Funcion stem corregida para que reciba un string en lugar de un objeto
#     import re

#     def stem(text):
#         y = []
#         for i in re.split(r'\W+', text):
#             y.append(ps.stem(i))
#         return " ".join(y)

#     similarity=cosine_similarity(vectors)


#     # Obtiene el índice de la película en el DataFrame new_df
#     movie_index = dfr1[dfr1['title'] == movie].index[0]
#     # Obtiene el género de la película a partir de su índice
#     movie_genre = dfr1.iloc[movie_index]['genres']

#     #Calcula las distancias entre la película y todas las demás películas en el DataFrame
#     distances = similarity[movie_index]
#     # Ordena las distancias en orden descendente y obtiene una lista de tuplas (índice, distancia)
#     movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]

#     #Inicializa un contador para llevar la cuenta de cuántas películas se han mostrado
#     count = 0
#     # Lista para almacenar las películas recomendadas
#     recommended_movies = []
    
#     # Recorre la lista de películas recomendadas
#     for i in movies_list:
#         # Verifica si la película recomendada tiene el mismo género que la película original
#         if dfr1.iloc[i[0]]['genres'] == movie_genre:
#             # Si es así, agrega el título de la película recomendada a la lista
#             recommended_movies.append(dfr1.iloc[i[0]].title)
#             # Incrementa el contador
#             count += 1
#             # Verifica si ya se han mostrado cinco películas
#             if count == 5:
#                 # Si es así, detiene el bucle
#                 break
    
#     return {
#         "mensaje": f"Películas recomendadas: {recommended_movies}"
#     }




