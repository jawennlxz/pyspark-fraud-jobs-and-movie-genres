"""Movie genre co-occurrence with PySpark RDDs (MovieLens 100K).

Joins 100,000 ratings with each movie's genre list, then counts how often
every pair of genres appears together across rated movies and lists the
top 3 co-occurring genres for each genre.

Run: spark-submit genre_cooccurrence.py
Expects the MovieLens 100K files in the same folder, renamed as
mov_rating.dat (u.data), mov_genre.dat (u.genre) and mov_item.dat (u.item).
"""
import os

from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("ECA_Q3").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
base = os.path.dirname(os.path.abspath(__file__))

# Ratings: user ID, movie ID, rating, timestamp (tab-separated)
rating_rdd = (sc.binaryFiles(os.path.join(base, "mov_rating.dat"))
              .flatMap(lambda kv: kv[1].decode("utf-8").splitlines())
              .map(lambda line: line.split("\t"))
              .map(lambda p: (int(p[1]), (int(p[0]), int(p[2])))))
print("Rating RDD (movie ID, (user ID, rating)):")
print(rating_rdd.take(5))
print("Number of ratings:", rating_rdd.count())

# Genres: genre name | genre ID
genre_rdd = (sc.binaryFiles(os.path.join(base, "mov_genre.dat"))
             .flatMap(lambda kv: kv[1].decode("utf-8").splitlines())
             .map(lambda line: line.split("|"))
             .map(lambda p: (int(p[1]), p[0])))
genre_names = sc.broadcast(genre_rdd.collectAsMap())
print("Genre RDD (genre ID, genre name):")
print(genre_rdd.sortByKey().collect())
print("Number of genres:", genre_rdd.count())


def parse_item(line):
    """Movie line: ID | title | release date | video date | URL | 19 genre flags."""
    p = line.split("|")
    genres = [genre_names.value[i] for i, flag in enumerate(p[5:24]) if flag == "1"]
    return (int(p[0]), (p[1], p[2], genres))


item_rdd = (sc.binaryFiles(os.path.join(base, "mov_item.dat"))
            .flatMap(lambda kv: kv[1].decode("latin-1").split("\n"))
            .filter(lambda line: line.strip() != "")
            .map(parse_item))
print("Item RDD (movie ID, (title, release date, genre list)):")
for row in item_rdd.take(3):
    print(row)
print("Number of movies:", item_rdd.count())

# Join each rating with its movie's genre list
movie_genre_rdd = item_rdd.map(lambda x: (x[0], x[1][2]))
rating_genre_rdd = rating_rdd.join(movie_genre_rdd)
print("Joined RDD (movie ID, ((user ID, rating), genre list)):")
for row in rating_genre_rdd.take(3):
    print(row)
print("Number of records after the join:", rating_genre_rdd.count())

# Count every ordered pair of different genres, then keep the top 3 per genre
pair_counts = (rating_genre_rdd
               .flatMap(lambda r: [((x, y), 1) for x in r[1][1] for y in r[1][1] if x != y])
               .reduceByKey(lambda a, b: a + b))
top3 = (pair_counts.map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
        .groupByKey()
        .mapValues(lambda v: sorted(v, key=lambda t: (-t[1], t[0]))[:3])
        .sortByKey()
        .collect())
print("Top 3 co-occurring genres for each genre ((genre X, genre Y), occurrence count):")
for genre_x, partners in top3:
    for genre_y, count in partners:
        print(((genre_x, genre_y), count))
with_pairs = {g for g, _ in top3}
print("Genres with no co-occurring genre:", sorted(set(genre_names.value.values()) - with_pairs))

spark.stop()
