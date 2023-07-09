#uvicorn main:app --reload
#http://127.0.0.1:8000/docs

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


import pandas as pd
from datetime import datetime

import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()


#-----------------------------------------

df = pd.read_csv('movies_dataset.csv')
movies = df
movies = df.loc[:, ['title', 'genres', 'overview']]
def convert(obj): 
    if isinstance(obj, str) and '{' in obj:
        L=[]
        for i in ast.literal_eval(obj):
            L.append(i['name']);
        return L
    
def convert2(obj): 
    if isinstance(obj, str) and '{' in obj:
        dic = ast.literal_eval(obj)
        return dic['name']
    
df['genres'] = df['genres'].apply(convert)
df['belongs_to_collection'] = df['belongs_to_collection'].apply(convert2)
df['production_companies'] = df['production_companies'].apply(convert)
df['production_countries'] = df['production_countries'].apply(convert)
df['spoken_languages'] = df['spoken_languages'].apply(convert)

# Rellenar los valores nulos con 0
df[['revenue', 'budget']] = df[['revenue', 'budget']].fillna(0)

df['release_year'] = df['release_date'].str.slice(0, 4)

df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d', errors='coerce')

# Eliminamos las filas con valores nulos en la columna "release_date"
df = df.dropna(subset=['release_date'])

df = df.drop(['video', 'imdb_id', 'adult', 'original_title', 'vote_count', 'poster_path', 'homepage'], axis=1)

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['return'] = df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] != 0 else 0, axis=1)


#FUNCION PELICULAS POR IDIOMA

@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma: str):
    # Filtrar las filas del dataframe que corresponden al idioma especificado
    peliculas_idioma = df[df['original_language'] == idioma]

    # Obtener la cantidad de películas producidas en ese idioma
    cantidad_peliculas = len(peliculas_idioma)

    return {'idioma':idioma, 'cantidad':cantidad_peliculas}
 
#FUNCION PELICULAS POR DURACION

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    # Filtrar las filas del dataframe que corresponden al título especificado
    titulo = df[df['title'] == pelicula]

    if len(titulo) > 0:
        # Obtener la duración y el año de lanzamiento de la película
        duracion = titulo['runtime'].values[0]
        anio = titulo['release_year'].values[0]

        #return duracion, anio
        return {'pelicula':pelicula, 'duracion':duracion, 'anio':anio}
    else:
        # Si no se encuentra la película, retornar valores por defecto o None
        #return None, None
        return {'pelicula':pelicula, 'duracion':None, 'anio':None}



#FUNCION PELICULAS POR PAIS

@app.get("/peliculas_pais/{pais}")
def peliculas_pais(pais: str):
    count = 0
    for countries in df['production_countries']:
        if countries is not None and pais in countries:
            count += 1
    #return {'cantidad de peliculas por pais': count}
    return {'pais':pais, 'cantidad':count}

#FUNCION PELICULAS POR PRODUCTORA

@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    revenue = 0
    movie_count = 0
    for companies, rev in zip(df['production_companies'], df['revenue']):
        if companies is not None and productora in companies:
            revenue += rev
            movie_count += 1
    #return {'productora':productora, 'ganancia_total':revenue, 'cantidad':movie_count}  
    return {'productora':productora, 'revenue_total': revenue,'cantidad':movie_count}          
              
#FUNCION FRANQUICIA

@app.get("/franquicia/{nombre_franquicia}")
def franquicia(nombre_franquicia: str):
    # Función lambda para sumar la ganancia de las películas de la franquicia
    sumar_ganancia = lambda x: x['revenue'] if nombre_franquicia in str(x['belongs_to_collection']) else 0

    # Filtrar las filas que corresponden a la franquicia
    franquicia_df = df[df['belongs_to_collection'].apply(lambda x: nombre_franquicia in str(x))]

    # Calcular la ganancia total, promedio y cantidad de películas
    ganancia_total = franquicia_df.apply(sumar_ganancia, axis=1).sum()
    ganancia_promedio = franquicia_df.apply(sumar_ganancia, axis=1).mean()
    cant_peliculas = len(franquicia_df)
    
    return {'franquicia':nombre_franquicia, 'cantidad':cant_peliculas, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}

    
#SISTEMA DE RECOMENDACION

@app.get("/recomendacion/{title}")
def recomendacion(title: str):
    #movies = pd.read_csv('movies_dataset.csv')
  #Filtra por genero
  # Buscar la película por título
    dfr = movies[movies['title'].str.lower() == title.lower()]

    # Si no se encuentra la película, devuelve un mensaje de error
    if dfr.empty:
        return "No se encontró la película"

    # Obtener la lista de géneros de la película
    genre_list = dfr.iloc[0]['genres'].split(',')

    # Filtrar el dataframe por el género de la película
    metadata = movies[movies['genres'].apply(lambda x: any(item for item in genre_list if item in x.split(',')))]
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    metadata['overview'] = metadata['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()   
    idx = indices[title]   
    sim_scores = list(enumerate(cosine_sim[idx]))    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)   
    sim_scores = sim_scores[1:6]   
    movie_indices = [i[0] for i in sim_scores]    
    mi_lista = metadata['title'].iloc[movie_indices].values.tolist()
    #return mi_lista
    return {'lista recomendada': mi_lista}