#librerias
import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import a_funciones as fn
import joblib

#### conectar_base_de_Datos

conn=sql.connect('db_movies.db')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
###### 1.Sistemas basados en popularidad ######
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#


#Top 10 peliculas más calificadas(mas vistas) 
mt= pd.read_sql('''select title, count(*) as vistas
            from full_tabla
            group by title
            order by vistas desc limit 10''', conn)

dfmt  = go.Bar( x=mt.title,y=mt.vistas, text=mt.vistas, textposition="outside")
Layout=go.Layout(title="Top 10 movies",xaxis={'title':'Pelicula'},yaxis={'title':'# Calificaciones'})
go.Figure(dfmt,Layout)

#Top 10 peliculas mas vistas y su calificacion

pd.read_sql('''select title, count(*) as vistas,avg(rating) as calificacion_promedio
            from full_tabla
            group by title
            order by vistas desc limit 10''', conn)



#Peliculas Mejor Calificadas con mas de 50 calificaciones

pd.read_sql("""with temp as (select title, 
            avg(rating) as avg_rat,
            count(*) as qty
            from full_tabla
            group by title
            order by avg_rat desc)
            select * from temp 
            where qty >50
            limit 10
            """, conn)

#Peliculas mejor calificadas por Año de estreno

moviey=pd.read_sql('select movyear from full_tabla', conn )
def años(año = list(np.sort(moviey['movyear'].unique(),kind="quicksort")[::-1])):
     texto= f"""select movyear, title,
            avg(rating) as promedio_cal,count(rating) as qty_cal
            from full_tabla where movyear={año} group by movyear,title order by promedio_cal desc limit 20"""
     print(pd.read_sql(texto, sql.connect('db_movies.db')))

print(interact(años))

#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
######2.1 Sistema de recomendación basado en contenido un solo producto - Manual######
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#

#importamos la base de datos de sql
peliculas=pd.read_sql("select *  from movies_split",conn)
peliculas.info()
#convertimos el año de calificacion y el de estreno int
peliculas[["movyear"]]=peliculas[["movyear"]].astype("int")

#escalamos los años  de estreno

sc=MinMaxScaler()
peliculas[["movyear"]]=sc.fit_transform(peliculas[["movyear"]])

#eliminamos columnas que no se van a usar

peliculas_dummy1=peliculas.drop(columns=['movieId','title'])

#exportar esta base para usar en otro modelos
joblib.dump(peliculas_dummy1,"s_peliculas_dummy1") 

#similitud con una pelicula (toy story de 1995) usando correlacion
pelicula='Toy Story (1995)'
ind_peli=peliculas[peliculas['title']==pelicula].index.values.astype(int)[0] #extrae el index y lo convierte a entero
similar_pelis=peliculas_dummy1.corrwith(peliculas_dummy1.iloc[ind_peli,:],axis=1)   #se correlacionan todas las filas o libros de dum2 con la fila del libro de interes y todas sus columnas
similar_pelis=similar_pelis.sort_values(ascending=False)
top_similar_pelis=similar_pelis.to_frame(name="correlación").iloc[0:11,] 
top_similar_pelis['title']=peliculas["title"] 
    

#simulitud de una pelicula desde una lista interactiva con las 10 mas parecidas
def recomendacion(pelis = list(peliculas["title"])):
     
     ind_peli=peliculas[peliculas['title']==pelis].index.values.astype(int)[0] #extrae el index y lo convierte a entero
     similar_pelis=peliculas_dummy1.corrwith(peliculas_dummy1.iloc[ind_peli,:],axis=1)   #se correlacionan todas las filas o libros de dum2 con la fila del libro de interes y todas sus columnas
     similar_pelis=similar_pelis.sort_values(ascending=False)
     top_similar_pelis=similar_pelis.to_frame(name="correlación").iloc[0:11,] 
     top_similar_pelis['title']=peliculas["title"] #empareja index con titulo de pelicula
    
     return top_similar_pelis


print(interact(recomendacion))


##se ve en los resultados que la correlacion depende mucho del año de estreno, tiende a acercar peliculas con año similar
###se intenta con una correlacion sin el año de estreno

peliculas_dummy2=peliculas_dummy1.drop(columns=['movyear'])

#similitud con una pelicula (toy story de 1995)
pelicula='Toy Story (1995)'
ind_peli=peliculas[peliculas['title']==pelicula].index.values.astype(int)[0] #extrae el index y lo convierte a entero
similar_pelis=peliculas_dummy2.corrwith(peliculas_dummy2.iloc[ind_peli,:],axis=1)   #se correlacionan todas las filas o libros de dum2 con la fila del libro de interes y todas sus columnas
similar_pelis=similar_pelis.sort_values(ascending=False)
top_similar_pelis=similar_pelis.to_frame(name="correlación").iloc[0:11,] 
top_similar_pelis['title']=peliculas["title"] 
    
#simulitud de una pelicula desde una lista interactiva con las 10 mas parecidas
def recomendacion(pelis = list(peliculas["title"])):
     
     ind_peli=peliculas[peliculas['title']==pelis].index.values.astype(int)[0] #extrae el index y lo convierte a entero
     similar_pelis=peliculas_dummy2.corrwith(peliculas_dummy2.iloc[ind_peli,:],axis=1)   #se correlacionan todas las filas o libros de dum2 con la fila del libro de interes y todas sus columnas
     similar_pelis=similar_pelis.sort_values(ascending=False)
     top_similar_pelis=similar_pelis.to_frame(name="correlación").iloc[0:11,] 
     top_similar_pelis['title']=peliculas["title"] #empareja index con titulo de pelicula
    
     return top_similar_pelis


print(interact(recomendacion))

#se ve mas varidad en los años de estreno de las peliculas similares, es una muestra mas variada.
#exportar esta base para usar en otro modelos
joblib.dump(peliculas_dummy2,"s_peliculas_dummy2") 


#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
######2.2 Sistema de recomendación basado en un solo producto- KNN ######
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#


## se entrena el modelo knn con la base de datos con año de estreno
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(peliculas_dummy1)
dist, idlist = model.kneighbors(peliculas_dummy1)

distancias=pd.DataFrame(dist) #distancias de las 10 peliculas mas cercanas
id_list=pd.DataFrame(idlist) #se traduen las distancias al id de la pelicula que corresponde

#modelo aplicado a cualquier pelicula
def MovieRecommender(movie_name = list(peliculas['title'].value_counts().index)):
    list_name = []
    movie_id = peliculas[peliculas['title'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:
        list_name.append(peliculas.loc[newid].title)
    return list_name


print(interact(MovieRecommender))

## se entrena el modelo knn con la base de datos SIN año de estreno
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(peliculas_dummy2)
dist, idlist = model.kneighbors(peliculas_dummy2)

distancias=pd.DataFrame(dist) #distancias de las 10 peliculas mas cercanas
id_list=pd.DataFrame(idlist) #se traduen las distancias al id de la pelicula que corresponde

#modelo aplicado a cualquier pelicula
def MovieRecommender(movie_name = list(peliculas['title'].value_counts().index)):
    list_name = []
    movie_id = peliculas[peliculas['title'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:
        list_name.append(peliculas.loc[newid].title)
    return list_name


print(interact(MovieRecommender))

#como en el modelo en el que se uso correlacion, se ve mayor variedad en los años d ela recomendaciones, pero las listas son mas similares



