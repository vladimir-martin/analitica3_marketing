import sqlite3 as sql 

import pandas as pd

### llevar bases de datos a sql

conn= sql.connect("db_movies.db") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.

### mostrar tablas dentro de la base de datos movies
cursor=conn.cursor() 
cursor.execute("select name from sqlite_master where type='table'")
cursor.fetchall()

#pregunta 1 registros que tiene la base de datos movies
#opcion 1
pd.read_sql("select count (movieId) as total from movies  ",conn)#si en movieId hubiera nulos, no los cuenta
#opcion 2 
pd.read_sql("""select count(*) from movies""", conn)

#pregunta 2 Consulta que muestre el número de usuarios han calificado películas en la plataforma
pd.read_sql("SELECT COUNT(DISTINCT userId) AS valores_unicos FROM ratings  ",conn)


#pregunta 3 Consulta que muestre el promedio de rating de la película con Id = 1
#opcion 1
pd.read_sql("select movieId, AVG(RATING) AS PROMEDIO_1 FROM ratings WHERE movieId=1",conn)
#opcion 2
pd.read_sql("""select movieId, avg(rating)
            from ratings
            where movieId=1
            group by movieId order by userId asc""", conn)

#pregunta 4 Consulta que Muestre la lista de nombres de las películas que  no tienen  evaluaciones (ratings).

pd.read_sql("""select a.title, count(b.rating) as cnt
            from movies a left join ratings b on a.movieId=b.movieId 
            group by a.title where b.rating is null order by cnt asc """, conn)

#preugnta 5 Consulta que Muestre la lista de nombres de las películas que  tienen 1 evaluación (ratings).
# opcion 1 sin nombre
pd.read_sql("select movieId, count(*) as num_votos from ratings group by movieId  having num_votos= 1  ",conn)

#opcion 2 con nombre
pd.read_sql("""select a.movieId, a.title, count(b.rating) as cnt
            from movies a left join ratings b on a.movieId=b.movieId 
            group by a.title having cnt=1 order by cnt asc """, conn)

#Pregunta 6 Consulta qué muestre el grupo de géneros, ocupa el puesto 9 de los que tienen más películas.
#opcion 1
pd.read_sql("""select genres, count(*) as cnt
            from movies 
            group by genres 
            order by cnt desc limit 8,1  """, conn)
#opcion 2
pd.read_sql("""with t1 as (select genres, count(*) as cnt 
            from movies 
            group by genres 
            order by cnt desc limit 9) select * from t1 order by cnt asc limit 1 """, conn)

#rating promedio por usuario
pd.read_sql("""select userId, avg(rating)
            from ratings
            group by userId order by userId asc""", conn)
