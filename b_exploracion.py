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

###ver caracteristicas de la base

movies.head()
ratings.head()



#separar generos en movies
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)
movies_split=pd.concat([movies,genres],axis=1)