select * from movies;
select * from ratings
DROP TABLE IF EXISTS base_full;
CREATE TABLE base_full AS 
SELECT t1.*, t2.*
FROM movies t1
LEFT JOIN ratings t2 ON t1.movieId=t2.movieId;


select * from base_full

--pregunta 1 registros que tiene la abse de datos movies
select count (movieId) as total from movies;

--pregunta 2 numero de usuarios que han calificado peliculas en la plataforma
SELECT COUNT(DISTINCT userId) AS valores_unicos
FROM ratings;

-- pregunta 3 promedio de rating de la pelicula con id=1
--opcion 1
select AVG(RATING) AS PROMEDIO_1 FROM ratings WHERE movieId=1

--opcion 2
select movieId, avg(rating)
            from ratings
            where movieId=1
            group by movieId order by userId asc

--pregunta 4 lista de nombres de las peliculas que no tienen  evaluaciones (ratings)
select a.title, count(b.rating) as cnt
from movies a left join ratings b on a.movieId=b.movieId 
where b.rating is null 
group by a.title
order by cnt asc;

-- preugnta 5 lista de nombres de peliculas con 1 evaluacion (ratings)
--opcion 1 sin nombre
select movieId, count(*) as num_votos from ratings
group by movieId 
having num_votos= 1

--opcion 2 con nombre
select a.movieId, a.title, count(b.rating) as cnt
            from movies a left join ratings b on a.movieId=b.movieId 
            group by a.title having cnt=1 order by cnt asc


--pregunta 6 
--opcion 1
select genres, count(*) as cnt 
            from movies 
            group by genres 
            order by cnt desc limit 9
--opcion 2
select genres, count(*) as cnt
            from movies 
            group by genres 
            order by cnt desc limit 8,1