from flask import Flask, render_template, request
import pandas as pd
import requests
import json

app = Flask(__name__)

# load your movie dataset into a pandas dataframe
df = pd.read_csv("/Users/ryanbirkedal/Documents/MovieList.csv", header=0)


# define a function to retrieve the movie poster from the TMDB API
def get_movie_poster(movie_id):
    imdb_id = 'tt' + str(movie_id)
    api_key = '6ea6649c788824ba87dc8a69da582999'
    imdb = 'https://api.themoviedb.org/3/find/{0}?api_key={1}&language=en-US&external_source=imdb_id'.format(
        imdb_id, api_key)
    rs = requests.get(url=imdb)
    jsonDataStreaming = json.loads(rs.text)
    poster = None  # default value
    if len(jsonDataStreaming['movie_results']) > 0 and jsonDataStreaming['movie_results'][0]['poster_path'] != None:
        poster = "https://image.tmdb.org/t/p/w500/" + jsonDataStreaming['movie_results'][0]['poster_path']
    return poster

# Define functions to filter and sort movies
def get_top_movies(n=10):
    return df.sort_values(by=['rating'], ascending=False).head(n)

def get_bottom_movies(n=10):
    return df.sort_values(by=['rating'], ascending=True).head(n)

def get_2022_movies(n=10):
    return df[df['year'] == 2022].sort_values(by=['rating'], ascending=False).head(n)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_query = request.form["search_query"]
        # filter the dataframe based on the search query
        filtered_df = df[df["Title"].str.contains(search_query, case=False)]
        # create a list of dictionaries to pass to the template
        movies = []
        for index, row in filtered_df.iterrows():
            movies.append({
                "title": row["Title"],
                "poster": get_movie_poster(row["Movie ID"]),
                "id": row["Movie ID"],
                "rating": str(row['My Rating'])
            })
        return render_template("results.html", movies=movies)
    else:
        # get the top reviewed movies from the dataframe
        top_movies_df = df.sort_values(by=["My Rating"], ascending=False).head(10)
        # get the top reviewed movies of 2022
        top_movies_2022_df = df[df["Year"] == 2022].sort_values(by=["My Rating"], ascending=False).head(10)
        # get the bottom reviewed movies from the dataframe
        bottom = df[df['My Rating'] > 0]
        bottom_movies_df = bottom.sort_values(by=["My Rating"], ascending=True).head(10)
        # create a list of dictionaries to pass to the template
        top_movies = []
        for index, row in top_movies_df.iterrows():
            top_movies.append({
                "title": row["Title"],
                "poster": get_movie_poster(row["Movie ID"]),
                "id": row["Movie ID"],
                "rating": str(int(row['My Rating']))
            })
        top_movies = pd.DataFrame(top_movies)
        # create a list of dictionaries to pass to the template
        top_movies_2022 = []
        for index, row in top_movies_2022_df.iterrows():
            top_movies_2022.append({
                "title": row["Title"],
                "poster": get_movie_poster(row["Movie ID"]),
                "id": row["Movie ID"],
                "rating": str(int(row['My Rating']))
            })
        top_movies_2022 = pd.DataFrame(top_movies_2022)
        # create a list of dictionaries to pass to the template
        bottom_movies = []
        for index, row in bottom_movies_df.iterrows():
            bottom_movies.append({
                "title": row["Title"],
                "poster": get_movie_poster(row["Movie ID"]),
                "id": row["Movie ID"],
                "rating": str(int(row['My Rating']))
            })
        bottom_movies = pd.DataFrame(bottom_movies)
        return render_template("index.html", top_movies=top_movies, top_movies_2022=top_movies_2022, bottom_movies=bottom_movies, get_movie_poster=get_movie_poster)



@app.route("/movie/<int:movie_id>")
def movie_details(movie_id):
    # retrieve the movie details from the dataframe
    movie_details = df[df["movie_id"] == movie_id].iloc[0]
    # get the movie poster using the TMDB API
    movie_poster = get_movie_poster(movie_details["Movie ID"])
    return render_template("movie_details.html", title=movie_details["Title"], poster=movie_poster, overview=movie_details["overview"])


if __name__ == "__main__":
    app.run(debug=True)
