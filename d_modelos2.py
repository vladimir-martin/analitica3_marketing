#librerias
import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import train_test_split

###conectar_base_de_Datos

conn=sql.connect('db_movies.db')
cur=conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
###### 3. Sistema de recomendación basado en contenido KNN ####
########## con base en todo lo visto por el usuario ###########
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#

movies=pd.read_sql("select * from movies_split",conn)
movies.info()# ver caracteristicas
#convertir a entero los años
movies[['movyear']]=movies[['movyear']].astype('int')

#cargar df escalado, con dummies CON año de estreno

peliculas_dummy1=joblib.load('s_peliculas_dummy1')

#que usuario voy a usar para mostrarle recomendaciones?
usuarios=pd.read_sql('select distinct (userId) as user_id from full_tabla',conn)

# funcion que permite seleccionar el usuario y muestra peliculas recoemndadas, segun lo que el ha visto
def recomendar1(user_id=list(sorted(usuarios['user_id'].value_counts().index))):
    
    ##seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select * from ratings_split where userId=:user',conn, params={'user':user_id})

    ###convertir peliculas del usuario a array
    user_movieId =ratings['movieId'].to_numpy()

    ###agregar la columna de movie id y titulo  a dummie para filtrar y mostrar nombre
    peliculas_dummy1[['movieId','title']]=movies[['movieId','title']]

    ### filtrar peliculas calificadas por el usuario
    movies_rated=peliculas_dummy1[peliculas_dummy1['movieId'].isin(user_movieId)]

    ## eliminar columna de id y titulo de pelicula
    movies_rated=movies_rated.drop(columns=['movieId','title'])
    movies_rated["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    ##centroide o perfil del usuario
    centroide=movies_rated.groupby("indice").mean() # sale del promedio de los valores de las peliculas vistas
    
    
    ### filtrar peliculas que no se ha visto
    movies_no_rated=peliculas_dummy1[~peliculas_dummy1['movieId'].isin(user_movieId)]
    ## eliminar id y nombre de peliculas no vistas
    movies_no_rated=movies_no_rated.drop(columns=['movieId','title'])
    
    ### entrenar modelo con peliculas no vistas para que de estas me recomiende las 10 mas cercanas al centroide
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_no_rated)
    dist, idlist = model.kneighbors(centroide) #nos queda una lista de los index de los recomendados
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_movies=movies.loc[ids][['title','movieId']] #sacamos el titulo y el id de las peliculas, segun la lista de index
    leidos=movies[movies['movieId'].isin(user_movieId)][['title','movieId']]#de la base de peliculas, me muestra solo las que estan en la lista de peliculas vistas o calificadas
    
    return recomend_movies

print(interact(recomendar1))


#cargar df escalado, con dummies SIN año de estreno

peliculas_dummy2=joblib.load('s_peliculas_dummy2')

#que usuario voy a usar para mostrarle recomendaciones?
usuarios=pd.read_sql('select distinct (userId) as user_id from full_tabla',conn)

# funcion que permite seleccionar el usuario y muestra peliculas recoemndadas, segun lo que el ha visto
def recomendar2(user_id=list(sorted(usuarios['user_id'].value_counts().index))):
    
    ##seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select * from ratings_split where userId=:user',conn, params={'user':user_id})

    ###convertir peliculas del usuario a array
    user_movieId =ratings['movieId'].to_numpy()

    ###agregar la columna de movie id y titulo  a dummie para filtrar y mostrar nombre
    peliculas_dummy2[['movieId','title']]=movies[['movieId','title']]

    ### filtrar peliculas calificadas por el usuario
    movies_rated=peliculas_dummy2[peliculas_dummy2['movieId'].isin(user_movieId)]

    ## eliminar columna de id y titulo de pelicula
    movies_rated=movies_rated.drop(columns=['movieId','title'])
    movies_rated["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    ##centroide o perfil del usuario
    centroide=movies_rated.groupby("indice").mean() # sale del promedio de los valores de las peliculas vistas
    
    
    ### filtrar peliculas que no se ha visto
    movies_no_rated=peliculas_dummy2[~peliculas_dummy1['movieId'].isin(user_movieId)]
    ## eliminar id y nombre de peliculas no vistas
    movies_no_rated=movies_no_rated.drop(columns=['movieId','title'])
    
    ### entrenar modelo con peliculas no vistas para que de estas me recomiende las 10 mas cercanas al centroide
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_no_rated)
    dist, idlist = model.kneighbors(centroide) #nos queda una lista de los index de los recomendados
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_movies=movies.loc[ids][['title','movieId']] #sacamos el titulo y el id de las peliculas, segun la lista de index
    leidos=movies[movies['movieId'].isin(user_movieId)][['title','movieId']]#de la base de peliculas, me muestra solo las que estan en la lista de peliculas vistas o calificadas
    
    return recomend_movies

print(interact(recomendar2))

#Al tener en cuenta el año de estreno, el modelo recomienda al usuario peliculas con años de estreno mas proximos que cuando no se tiene en cuenta. 



#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
###### 4. Sistema de recomendación de filtro colaborativo #####
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#
#¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬#

#traemos base de datos de calificaciones (no tenemos calificaciones nulas o en 0)
ratings=pd.read_sql("select * from ratings_split",conn)
#eliminamos columna que no necesitamos para el modelo
ratings=ratings.drop(columns=['rating_time'])

#configurar escala de surprise
np.sort(ratings["rating"].unique()) #vemos una escala de 0-5
reader = Reader(rating_scale=(0, 5))
#se carga la data en el orden estandar de columnas
data   = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

#se crea lista de modelos a evaluar
models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline()] 
results = {}


##se van a probar todos modelos y ver cual tiene mejor desempeño (supervisado, porque se quiere predecir la calificacion que se le dara a una pelicula que no se ha visto)

for model in models:
 
    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
    
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result

# resultados de las metricas de los algoritmos
performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')

##los indicadores muestran que el mejor modelo es el KNN BASELINE, ya que tiene el RMSE y MAE mas bajos. 
##en este caso se da prioridad al desempeño que a los tiempos de procesamiento.


###se selecciona KNN Baseline ###
param_grid = { 'sim_options' : {'name': ['msd','cosine'], \
                                'min_support': [5], \
                                'user_based': [False, True]}#este selecciona solo si es basado en usuario o item
             }
## se hace afinamiento de hiperparametros
gridsearchKNNBaseline = GridSearchCV(KNNBaseline, param_grid, measures=['rmse'], \
                                      cv=4, n_jobs=2)
                                    
gridsearchKNNBaseline.fit(data)

gridsearchKNNBaseline.best_params["rmse"]  #gano basado en item 
gridsearchKNNBaseline.best_score["rmse"]
#mejor modelo
gs_model=gridsearchKNNBaseline.best_estimator['rmse'] #no es necesario volver a entrenar, este best estimator esta ya entrenado con los hiperparametros ganadores.



###Entrenar con todos los datos en train y Realizar predicciones con el modelo afinado

trainset = data.build_full_trainset() 
model=gs_model.fit(trainset) # se vuelve a entrenar, pero con todos los datos como train, en el anterior estaban divididos train/test

predset = trainset.build_anti_testset() ### crea una tabla con todos los usuarios y sus respectivas peliculas no vistas, ya que todo se ha entrenado con los usuarios y las peliculas vistas y calificadas por ellos
len(predset)

predictions = gs_model.test(predset) ### hace las predicciones del rating para todas las peliculas no vistas por cada usuario, ya no es .fit, es .test
### la funcion test recibe un test set construido con build_test method, o el que genera crosvalidate

predictions_df = pd.DataFrame(predictions)
predictions_df.shape
predictions_df.head()
predictions_df['r_ui'].unique() ### promedio de ratings
predictions_df.sort_values(by='est',ascending=False)


###funcion que recomienda las 10 peliculas con mejor prediccion por usuario y exporta la base de datos a sql para consultar mas adelante
def recomendaciones(user_id,n_recomend=10):
    
    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid','est']]
    recomendados.to_sql('reco',conn,if_exists="replace",index=False)
    
    recomendados=pd.read_sql('''select a.*, b.title 
                             from reco a left join movies_split b
                             on a.iid=b.movieId ''', conn)

    return(recomendados)


 
recomendaciones(user_id=150,n_recomend=10)