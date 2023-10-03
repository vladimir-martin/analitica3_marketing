#importar librerias
import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go ### para gráficos
import plotly.express as px
import a_funciones as fn
from mlxtend.preprocessing import TransactionEncoder

#conexion con sql 

conn= sql.connect("db_movies.db") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.

### mostrar tablas dentro de la base de datos
cursor=conn.cursor() 
cursor.execute("select name from sqlite_master where type='table'")
cursor.fetchall()

###importar tablas de sql a python

movies=pd.read_sql("SELECT * FROM movies",conn)
ratings=pd.read_sql("SELECT * FROM ratings",conn)

###ver caracteristicas de las base

movies.head()#vision general
movies.info()#tipo de cada variable
movies.duplicated().sum()#veriicar duplicados

ratings.head()#vision general
ratings.info()#tipo de cada variable
ratings.duplicated().sum()#verificar duplicados

#Vemos si las tablas tienen valores nulos.
print(movies.isnull().sum())
print(ratings.isnull().sum())

#En el dataframe ratings se cambia el formato de timestamp por uno de fecha para hacer su manejo mas cómodo.
ratings['rating_time'] =pd.to_datetime(ratings['timestamp'], unit='s')
ratings.drop(columns=["timestamp"],inplace=True)
ratings.head()
#llevarla a sql
ratings.to_sql("ratings_split",conn,if_exists="replace",index=False)
pd.read_sql("""select * from ratings_split""",conn)



#separar generos en db movies
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)

#extraemos año de estreno del titulo
movies['movyear'] = movies['title'].str.strip().str[-5:-1]

#valores extraidos
movies['movyear'].unique()

#peliculas sin año de estreno especifico

movies[movies['movyear'].str.contains('[a-zA-Z]')] #info completa

sin_a=movies["movyear"][movies['movyear'].str.contains('[a-zA-Z]')]

#cantidad de calificaciones de estas peliculas
ratings[ratings["movieId"].isin(movies["movieId"][movies['movyear'].str.contains('[a-zA-Z]')])].groupby("movieId").size()


#poner en columna de año de estreno el valor "No_year" a peliculas sin año
#Funcion para remplazar valores de columna
def reemplazar_valor(valor):
  if valor in sin_a.values:
      return "0"
  return valor
movies["movyear"]=movies['movyear'].apply(reemplazar_valor)

#creamos nueva base con esta info
movies_split=pd.concat([movies,genres],axis=1).drop(["genres"],axis=1)
#llevarla a sql
movies_split.to_sql("movies_split",conn,if_exists="replace",index=False)
pd.read_sql("""select * from movies_split""",conn)

#verificar categorias por errores tipograficos o categorias similares
#generos de peliculas
genres.columns.tolist()
#ratings
np.sort(ratings["rating"].unique()) #no hay calificaciones de 0



##Analisis de categorias
#se pretende saber si hay categorias que por su distribucion no aporten a los objetivos de la solucion y deban ser recortadas

#base de datos de peliculas
# Cantidad de peliculas por genero (usando python, todos los generos al tiempo)
gen_total=pd.DataFrame(movies_split.drop(["title","movieId","movyear"],axis=1).sum()).reset_index()
gen_total.columns=["Genre","Qty"]
gen_total=gen_total.sort_values(by="Qty",ascending=False)

#grafico
fig  = px.bar(gen_total, x= 'Genre',y="Qty",
              title= 'Numero de peliculas por genero',
              labels={'Genre':'Genero'})
fig.show()

# Cantidad de peliculas por genero (usando sql, se debe especificar que genero)
gen_sum=pd.read_sql("""SELECT SUM(Comedy) AS Qty_Comedy, SUM(Action) AS Qty_Action, SUM(Animation) AS Qty_Animation
FROM movies_split""",conn)
gen_sum

#vemos que todos los generos aportan al analisis, con una cantidad de datos significativa por genero y eliminar los generos con pocas calificaciones, no aporta tampoco a la eficiencia del codigo.

#Base de datos de calificaciones
#cantidad de calificaciones por usuario
rating_user=pd.read_sql(''' select "userId" as User_Id,
                         count(*) as Qty
                         from ratings_split
                         group by "userId"
                         order by Qty asc
                         ''',conn )
fig  = px.histogram(rating_user, x= 'Qty', title= 'Hist frecuencia de numero de calificaciones por usuario')
fig.show() 
rating_user.describe()

#ver rango de fechas de las calificaciones

pd.read_sql("""select min(strftime('%Y',rating_time)) as año_min, max(strftime('%Y',rating_time)) as año_max,max(strftime('%Y',rating_time))-min(strftime('%Y',rating_time)) as rango_años from ratings_split""",conn)


# no vemos necesario eliminar usuarios, ya que el minimo de calificaciones por usuario es 20 que es un buen numero y solo un 25% de los usuarios tienen mas de 168 calificaciones, con datos atipicos que suben el promedio a 165 calificaciones por usuario.
#ademas vemos que el rango de las calificaciones es 22 años, tiempo en el cual un usuario puede calificar 2698 peliculas que es el valor maximo (aprox 10 peliculas mensuales)


#cantidad de calificaciones por pelicula
rating_movie=pd.read_sql(''' select "movieId" as Movie_Id,
                         count(*) as Qty
                         from ratings_split
                         group by "movieId"
                         order by Qty asc
                         ''',conn )
fig  = px.histogram(rating_movie, x= 'Qty', title= 'Hist frecuencia de numero de calificaciones por pelicula')
fig.show() 
rating_movie.describe()
#no vemos necesario eliminar peliculas, la mitad de ellas tienen menos de 3 calificaciones y solo un 25% tienen mas de nuevo, con datos atipicos de un maximo de 329 calificaciones que elevan el promedio a 10 calificaciones por pelicula.


#analisis adicionales

#Número de usuarios que calificaron títulos.
pd.read_sql("""select count(distinct(userId)) as cantidad_usuarios from ratings_split""", conn)



#tabla completa creada en sql
fn.ejecutar_sql('preprocesamiento.sql', cursor)

cursor.execute("select name from sqlite_master where type='table' ")
cursor.fetchall()

#tabla completa
pd.read_sql("""select * from full_tabla""",conn)