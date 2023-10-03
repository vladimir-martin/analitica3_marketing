#cargar paquetes
import pandas as pd 
import sqlite3 as sql 
import sys 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 

#Función que imputa datos para variables numéricas
def impute_columns(df, columns, strategy): 
  imputer = SimpleImputer(strategy=strategy)
  for column in columns:
    column_imputed = imputer.fit_transform(df[column].values.reshape(-1, 1))
    df[column] = column_imputed.flatten()
  return df

#Funcion que ejecuta sql en python        
def ejecutar_sql (nombre_archivo, cur):
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close
  cur.executescript(sql_as_string)

