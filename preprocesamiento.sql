drop table if exists full_tabla ;

create table full_tabla as select 
a.*,
b.userId,
b.rating,
b.rating_time
 from movies_split a 
 inner join
 ratings_split b on a.movieId=b.movieId;
 