import pandas as pd

title_data = pd.read_csv("imdb_data/title.akas.tsv/data.tsv", sep="\t")
us_data = title_data.loc[title_data.region == "US"]


us_data.index = us_data.titleId

join_data = pd.read_csv("imdb_data/title.basics.tsv/data.tsv", sep="\t",
                        dtype={"tconst":str, "titleType":str, "primaryTitle":str, "originalTitle":str, "isAdult":object, "startYes":object, "endYear":object, "runtimeMinutes":object, "genres":str})
join_data.index = join_data.tconst
us_data = us_data.join(join_data, how="left")

join_data = pd.read_csv("imdb_data/title.ratings.tsv/data.tsv", sep="\t",
                        dtype={"tconst":str, "averageRating":float, "numVotes":int})

join_data.index = join_data.tconst
us_data = us_data.join(join_data.drop(columns='tconst'), how="left")

us_data.to_csv("./imdb_data_post_processed/us_title.akas.csv", encoding="utf-8")