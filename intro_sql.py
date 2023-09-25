import sqlite3 as sql
import pandas as pd

#conectarse o crear  base de datos de sql desde drive
conn=sql.connect("/Users/Vlado/Documents/universidad/11_sem/Analitica3/caso_estudio_marketing/db_movies.db")

#crea cursor-ejecutar consulta en base de datos, se usan recursos del servidos
cur=conn.cursor()

#verificar que se conecto a base de datos, ejecutando sqlite_master, es la base con todas las tablas de la base de datos
#se esta ejeuctando en la base de datos
cur.execute("select name from sqlite_master where type='table'")# ver tablas de la base de datos
#ver resultado de la consulta
cur.fetchall()

#se crea dataframe de pandas con consulta de sql, dando la conn o la conexion a la db
df=pd.read_sql("select * from ratings",conn)
df

#llevar tablas de pandas  hacia sql
df.to_sql("ratings_copia",conn,if_exists="replace")

#crear tablas, con comillas tripes se puede hacer salto de linea
cur.execute("""DROP TABLE IF EXISTS ratings5""")# me borra la tabla para que no me genere conflicto con el nombre
cur.execute("""create table ratings5
 as select * from ratings
  where userId=1""") #el cur ejecuta la consulta en la base de datos sin traerla a python

#ver tabla creada de userId 1

pd.read_sql("select * from ratings5",conn)

#ver de neuvo estado de la basse
cur.execute("select name from sqlite_master where type='table'")# ver tablas de la base de datos
#ver resultado de la consulta
cur.fetchall()

help("modules")