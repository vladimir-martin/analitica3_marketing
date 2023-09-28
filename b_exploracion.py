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


#separar generos en db movies
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)
movies_split=pd.concat([movies,genres],axis=1).drop(["genres"],axis=1)
#llevarla a sql
movies_split.to_sql("movies_split",conn,if_exists="replace")

#verificar categorias por errores tipograficos o categorias similares
#generos de peliculas
genres.columns.tolist()
#ratings
np.sort(ratings["rating"].unique())

#base de datos de peliculas
# Cantidad de peliculas por genero
gen_total=pd.DataFrame(movies_split.drop(["title","movieId"],axis=1).sum()).reset_index()
gen_total.columns=["Genre","Qty"]
gen_total=gen_total.sort_values(by="Qty",ascending=False)
fig  = px.bar(gen_total, x= 'Genre',y="Qty",
              title= 'Numero de peliculas por genero',
              labels={'Genre':'Genero'})
fig.show()

#Base de datos de calificaciones
#cantidad de calificaciones por usuario
rating_user=pd.read_sql(''' select "userId" as User_Id,
                         count(*) as Qty
                         from ratings
                         group by "userId"
                         order by Qty asc
                         ''',conn )
fig  = px.histogram(rating_user, x= 'Qty', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show() 
rating_user.describe()
# no vemos necesario eliminar usuarios, ya que el minimo de calificaciones por usuario el 20 que es un buen numero y solo un 25% de los usuarios tienen mas de 168 calificaciones.




