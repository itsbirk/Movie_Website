from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import requests
from imdb import Cinemagoer
ia = Cinemagoer()
import json
from collections import defaultdict
from typing import Dict, Any
from surprise import Dataset, SVD, Reader
from surprise.model_selection import RandomizedSearchCV
import numpy as np
import textwrap
from PIL import Image
import plotly.express as px


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def display_home():
    if request.method == 'POST':
        query = request.form.get('query')
        return redirect(url_for('results', query=query))

    return render_template('home.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    API = '6ea6649c788824ba87dc8a69da582999'
    movieList = []
    query = request.form.get('query')
    movie_results = ia.search_movie(query)
    for movie in movie_results:
        imdbId = 'tt' + movie.__dict__['movieID']
        imdb = 'https://api.themoviedb.org/3/find/{0}?api_key={1}&language=en-US&external_source=imdb_id'.format(
            imdbId, API)
        rs = requests.get(url=imdb)
        jsonDataStreaming = json.loads(rs.text)
        if len(jsonDataStreaming['movie_results']) > 0 and jsonDataStreaming['movie_results'][0]['poster_path'] != None:
            poster = "https://image.tmdb.org/t/p/w500/" + jsonDataStreaming['movie_results'][0]['poster_path']
        else:
            poster = "No Poster"
        if 'year' in movie.__dict__['data']:
            year = str(movie["year"])
        else:
            year = "No Information"
        if len(jsonDataStreaming['movie_results']) > 0:
            plot = jsonDataStreaming['movie_results'][0]['overview']
            tmdb = jsonDataStreaming['movie_results'][0]['id']
        else:
            plot = "No Information"
            tmdb = 'none'
        title = movie['title']
        movieList.append({'title': title, 'poster': poster, 'year': year, 'plot': plot, 'tmdb': tmdb, 'ttimdb': imdbId})
    return render_template('results.html', movieList=movieList)


# Movie database page
@app.route("/database")
def display_movies():
    df = pd.read_csv("/Users/ryanbirkedal/Documents/MovieListList.csv", header=0)
    df = df.sort_values('My Rating', ascending=False)
    movies = []
    for _, movie in df.iterrows():
        # Convert the DataFrame row to a dictionary
        movie_dict = movie.to_dict()
        # Add additional keys and values to the dictionary
        movie_dict['rating'] = str(int(movie['My Rating'])) + '/' + '100' if movie['My Rating'] > 0 else 'No Rating'
        # Add the dictionary to the list of movies
        movies.append(movie_dict)

    return render_template('database.html', movies=movies)


# Movie algo page
@app.route("/recommendation")
def movie_recs():
    moviesdf = pd.read_csv('/Users/ryanbirkedal/Documents/TestSet.csv', header=0)

    moviesdf['year'] = moviesdf['year'].astype(str).str[:4]
    omdb = 'c0943f0'

    user_id = 31415926535
    streamingSites = ['Amazon Prime Video', 'Disney Plus', 'Max', 'Hulu', 'Netflix', 'Paramount Plus',
                      'Peacock Premium', 'Showtime', 'Apple TV Plus',
                      'DIRECTV', 'Hoopla', 'CBS', 'fuboTV', 'MGM Plus', 'AMC+',
                      'USA Network', 'TCM', 'FXNow', 'BroadwayHD', 'Criterion Channel',
                      'Film Movement Plus', 'Starz',
                      'Cinemax Amazon Channel', 'Sling TV Orange and Blue']

    genres = ['', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
              'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War']

    def runAlgo(userId, moviesData):
        min_rating = 0
        max_rating = 5
        reader = Reader(rating_scale=(min_rating, max_rating))
        df = pd.read_csv('/Users/ryanbirkedal/Documents/mediumratings.csv', header=0)
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

        trainset = data.build_full_trainset()

        n_factors = 100
        n_epochs = 40
        lr_all = 0.005
        reg_all = 0.02

        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        algo.fit(trainset)
        movieIDs = df['movieId'].unique()
        test = moviesData['movieId']
        movieIDs = np.intersect1d(movieIDs, test)
        userMovieIDs = df.loc[df['userId'] == userId, 'movieId']
        notWatched = np.setdiff1d(movieIDs, userMovieIDs)
        testSet = [(userId, movie, trainset.global_mean) for movie in notWatched]
        predictions = algo.test(testSet)

        pred_ratings = np.array([(str(pred.iid), pred.est) for pred in predictions])
        pred_est = pred_ratings[:, 1]  # select all the pred.est values
        indexOrder = np.argsort(pred_est)[::-1]  # sort in descending order and select top n_items
        sorted_pred_ratings = pred_ratings[indexOrder][:100]  # select the top n_items tuples from pred_ratings

        return sorted_pred_ratings

    def getStreamingServices(tmdb, API):
        streaming = []
        streamingSitesLink = 'https://api.themoviedb.org/3/movie/{0}/watch/providers?api_key={1}'.format(tmdb, API)
        rs = requests.get(url=streamingSitesLink)
        jsonDataStreaming = json.loads(rs.text)
        i = 0
        if 'results' in jsonDataStreaming and 'US' in jsonDataStreaming['results'] and 'flatrate' in \
                jsonDataStreaming['results']['US']:
            while i < len(jsonDataStreaming['results']['US']['flatrate']):
                if jsonDataStreaming['results']['US']['flatrate'][i]['provider_name'] in streamingSites:
                    streaming.append(jsonDataStreaming['results']['US']['flatrate'][i]['provider_name'])
                i += 1
        else:
            streaming.append('No streaming options')
        return streaming

    def createList(sorted, API, moviesdf):
        moviePreds = []
        for items in sorted:
            idRaw = int(items[0])
            score = round(float(items[1]), 2)
            print(score)
            title = moviesdf.loc[moviesdf['movieId'] == idRaw, 'title'].values[0]
            year = moviesdf.loc[moviesdf['movieId'] == idRaw, 'year'].values[0]
            tmdbID = moviesdf.loc[moviesdf['movieId'] == idRaw, 'tmdbId'].values[0]
            ttimdb = moviesdf.loc[moviesdf['movieId'] == idRaw, 'ttimdb'].values[0]
            estScore = score * 20
            if ttimdb != 'tt00000id':
                details = 'http://www.omdbapi.com/?i={0}&apikey={1}'.format(ttimdb, omdb)
                rd = requests.get(url=details)
                jsonData = rd.json()
                genress = jsonData['Genre']
                streamingOptions = getStreamingServices(tmdbID, API)
                director = jsonData['Director']
                cast = jsonData['Actors']
                plot = jsonData['Plot']
                poster = jsonData['Poster']
            else:
                genress = 'No Genre Information'
                streamingOptions = ['No Streaming Information']
                director = 'No Director Information'
                cast = 'No Cast Information'
                plot = 'No Plot Information'
                poster = "Pictures/movie.jpeg"
            moviePreds.append({'movieId': idRaw, 'ttimdb': ttimdb, 'tmdb': tmdbID, 'title': title,
                                           'year': year, 'estimate': items[1], 'genres': genress,
                                           'streaming': [streamingOptions], 'director': director,
                                           'cast': cast, 'plot': plot, 'poster': poster, 'estScore': estScore})
        return moviePreds

    sorted_pred_ratings = runAlgo(user_id, moviesdf)
    n_items = 10
    i = 0
    API = '6ea6649c788824ba87dc8a69da582999'
    moviePredictions = createList(sorted_pred_ratings, API, moviesdf)
    return render_template('recommendation.html', moviePredictions=moviePredictions)


@app.route('/movie/<string:ttimdb>')
def movie(ttimdb):
    movieInfo = []
    df = pd.read_csv("/Users/ryanbirkedal/Documents/MovieListList.csv", header=0)
    API = '6ea6649c788824ba87dc8a69da582999'
    omdb = 'c0943f0'
    imdb = 'https://api.themoviedb.org/3/find/{0}?api_key={1}&language=en-US&external_source=imdb_id'.format(ttimdb, API)
    re = requests.get(url=imdb)
    tmdbExternal = re.json()
    omdbInfo = 'http://www.omdbapi.com/?i={0}&apikey={1}'.format(ttimdb, omdb)
    rd = requests.get(url=omdbInfo)
    jsonData = rd.json()
    title = jsonData['Title']
    date = tmdbExternal['movie_results'][0]['release_date'][:4]
    rated = jsonData['Rated']
    overview = tmdbExternal['movie_results'][0]['overview']
    poster = "https://image.tmdb.org/t/p/w500/" + tmdbExternal['movie_results'][0]['poster_path']
    backDrop = "https://image.tmdb.org/t/p/w500/" + tmdbExternal['movie_results'][0]['backdrop_path']
    director = jsonData['Director']
    actors = jsonData['Actors']
    genre = jsonData['Genre']
    runtime = jsonData['Runtime']
    if ' ' in title:
        # Replace spaces with dashes
        rtTitle = title.replace(' ', '_')
        metaTitle = title.replace(' ', '-')
    if ttimdb in df['ttimdb'].values:
        if df.loc[df['ttimdb'] == ttimdb, 'My Rating'].values[0] > 0:
            rating = str(int(df.loc[df['ttimdb'] == ttimdb, 'My Rating'].values[0])) + '/' + '100'
        else:
            rating = "No Rating"
    else:
        rating = "No Rating"
    #Pictures on HTML side
    movieRankings = 'https://www.movierankings.net/review/{0}'.format(tmdbExternal['movie_results'][0]['id'])

    def get_rotten_tomatoes_rating(json_data):
        for rating in json_data['Ratings']:
            if rating['Source'] == 'Rotten Tomatoes':
                return rating['Value']
        return "No Rating"

    if jsonData['Ratings']:
        RT = get_rotten_tomatoes_rating(jsonData)
        if 'imdbRating' in jsonData:
            imdb = str(jsonData['imdbRating']) + '/10'
        else:
            imdb = 'No Rating'
        if 'Metascore' in jsonData:
            meta = str(jsonData['Metascore']) + '/100'
        else:
            meta = 'No Rating'
    else:
        RT = "No Rating"
        imdb = "No Rating"
        meta = "No Rating"

    tmdb = tmdbExternal['movie_results'][0]['id']
    streaming = []
    streamingSitesLink = 'https://api.themoviedb.org/3/movie/{0}/watch/providers?api_key={1}'.format(tmdbExternal['movie_results'][0]['id'], API)
    rs = requests.get(url=streamingSitesLink)
    jsonDataStreaming = json.loads(rs.text)
    streamingSites = ['Amazon Prime Video', 'Disney Plus', 'Max', 'Hulu', 'Netflix', 'Paramount Plus',
                      'Peacock Premium', 'Showtime', 'Apple TV Plus',
                      'DIRECTV', 'Hoopla', 'CBS', 'fuboTV', 'MGM Plus', 'AMC+',
                      'USA Network', 'TCM', 'FXNow', 'BroadwayHD', 'Criterion Channel',
                      'Film Movement Plus', 'Starz',
                      'Cinemax Amazon Channel', 'Sling TV Orange and Blue']
    nonSites = ['HBO Max Amazon Channel']
    i = 0
    if 'results' in jsonDataStreaming and 'US' in jsonDataStreaming['results'] and 'flatrate' in \
            jsonDataStreaming['results']['US']:
        while i < len(jsonDataStreaming['results']['US']['flatrate']):
            if jsonDataStreaming['results']['US']['flatrate'][i]['provider_name'] in streamingSites:
                streaming.append(jsonDataStreaming['results']['US']['flatrate'][i]['provider_name'])
            i += 1
    else:
        streaming.append('No streaming options')
    i = 0
    #Build out date last watched using index and dataframe and times watched - pass the index

    video = 'https://api.themoviedb.org/3/movie/{0}/videos?api_key={1}&language=en-US'.format(
        tmdbExternal['movie_results'][0]['id'], API)
    rv = requests.get(url=video)
    videoData = rv.json()
    trailers = []
    trailerNames = []
    for loops in videoData['results']:
        if loops['type'] == 'Trailer':
            trailer = 'https://www.youtube.com/embed/' + loops['key']
            trailers.append(trailer)
            trailerNames.append(loops['name'])

    movieInfo = {
        'title': title,
        'rtTitle': rtTitle,
        'metaTitle': metaTitle,
        'date': date,
        'rated': rated,
        'overview': overview,
        'poster': poster,
        'director': director,
        'actors': actors,
        'genre': genre,
        'runtime': runtime,
        'rating': rating,
        'movieRankings': movieRankings,
        'RT': RT,
        'imdb': imdb,
        'meta': meta,
        'streaming': streaming,
        'trailers': trailers,
        'trailerNames': trailerNames,
        'backDrop': backDrop,
        'ttimdb': ttimdb
    }

    return render_template('movie.html', movieInfo=movieInfo)


df = pd.read_csv("/Users/ryanbirkedal/Documents/MovieListList.csv", header=0)
df["Date Watched"] = pd.to_datetime(df["Date Watched"])



# Analytics page
@app.route('/analytics')
def analytics():

    return render_template('analytics.html')


@app.route('/histogram_data')
def histogram_data():
    year_counts = df['Year'].value_counts().sort_index()
    data = {"years": year_counts.index.tolist(), "counts": year_counts.values.tolist()}
    return jsonify(data)


@app.route('/bar_chart_data')
def bar_chart_data():
    df["Watched Year"] = df["Date Watched"].dt.year
    year_counts = df["Watched Year"].value_counts().sort_index()
    data = {"years": year_counts.index.tolist(), "counts": year_counts.values.tolist()}
    return jsonify(data)


@app.route('/top_movie_counts')
def top_movie_counts():
    peopleDFIn = pd.read_csv("/Users/ryanbirkedal/Documents/PeopleList.csv", header=0, converters={'Scores': pd.eval},
                             dtype={'Index ID': str, 'ID': str, 'Name': str, 'Role': str, 'Movie Count': int,
                                    'Movies Reviewed': int, 'Movies': str, 'Average Score': float})
    peopleDFIn.set_index('Index ID', inplace=True, drop=False)
    top_counts = peopleDFIn.groupby("Role").apply(lambda group: group.nlargest(10, "Movie Count")).reset_index(drop=True)
    top_counts = top_counts[["Name", "Role", "Movie Count"]]
    API = '6ea6649c788824ba87dc8a69da582999'

    directors = top_counts[top_counts["Role"] == "director"]
    actors = top_counts[top_counts["Role"] == "cast"]
    writers = top_counts[top_counts["Role"] == "writer"]
    faceList = []
    for name in directors['Name']:
        url = "https://api.themoviedb.org/3/search/person?query={0}&include_adult=false&api_key={1}&language=en-US&page=1".format(
            name, API)
        faces = requests.get(url=url)
        faces = faces.json()
        id = faces['results'][0]['id']
        url2 = 'https://api.themoviedb.org/3/person/{0}/images?api_key={1}'.format(id, API)
        faces2 = requests.get(url=url2)
        faces2 = faces2.json()
        pics = 'https://image.tmdb.org/t/p/w500' + faces2['profiles'][0]['file_path']
        faceList.append(pics)
    data = {
        "director_names": directors["Name"].tolist(),
        "director_pics": faceList,
        "director_counts": directors["Movie Count"].tolist(),
        #"actor_names": actors["Name"].tolist(),
        #"actor_counts": actors["Movie Count"].tolist(),
        #"writer_names": writers["Name"].tolist(),
        #"writer_counts": writers["Movie Count"].tolist(),
    }
    return jsonify(data)


@app.route('/top_avg_score_counts')
def top_avg_score_counts():
    peopleDFIn = pd.read_csv("/Users/ryanbirkedal/Documents/PeopleList.csv", header=0, converters={'Scores': pd.eval},
                             dtype={'Index ID': str, 'ID': str, 'Name': str, 'Role': str, 'Movie Count': int,
                                    'Movies Reviewed': int, 'Movies': str, 'Average Score': float})
    peopleDFIn.set_index('Index ID', inplace=True, drop=False)
    filtered_df = peopleDFIn[peopleDFIn["Movies Reviewed"] > 2]
    top_avg_score_counts = filtered_df.groupby("Role").apply(lambda group: group.nlargest(10, "Average Score")).reset_index(drop=True)
    directors = top_avg_score_counts[top_avg_score_counts["Role"] == "director"]
    actors = top_avg_score_counts[top_avg_score_counts["Role"] == "cast"]
    writers = top_avg_score_counts[top_avg_score_counts["Role"] == "writer"]
    data = {
        "director_names": directors["Name"].tolist(),
        "director_counts": directors["Average Score"].tolist(),
        "actor_names": actors["Name"].tolist(),
        "actor_counts": actors["Average Score"].tolist(),
        "writer_names": writers["Name"].tolist(),
        "writer_counts": writers["Average Score"].tolist(),
    }
    return jsonify(data)


@app.route('/movie/addForm/<string:ttimdb>', methods=['POST'])
def addForm(ttimdb):
    myRating = float(request.form.get('score'))
    dateWatched = request.form.get('date')
    myReview = request.form.get('review')
    ttimdb = ttimdb
    imdb = ttimdb[2:]
    movie = ia.get_movie(imdb)
    peopleDFIn = pd.read_csv("/Users/ryanbirkedal/Documents/PeopleList.csv", header=0, converters={'Scores': pd.eval},
                             dtype={'Index ID': str, 'ID': str, 'Name': str, 'Role': str, 'Movie Count': int,
                                    'Movies Reviewed': int, 'Movies': str, 'Average Score': float})
    movieIndexID = movie['title'][0] + movie['imdbID'] + movie['title'][0]  # need to change
    movieCriteria = ['production companies', 'director', 'writer', 'cinematographer', 'cast',
                     'genres', 'composer', 'editor', 'producer', 'costume designer', 'make up',
                     'art direction', 'production design', 'set decoration']
    searchablePeople = ['cast', 'director', 'writer', 'cinematographer', 'composer', 'genres', 'production companies',
                        'editor', 'producer', 'costume designer', 'make up', 'art direction', 'production design',
                        'set decoration']
    movieTitle = movie['title'].replace(',', '')

    if movieIndexID in df.index:
        # append score if new and run updatePeople, append review if there is review, and append watch date
        df.at[movieIndexID, 'My Rating'].append()
        df.at[movieIndexID, 'Date Watched'].append()
        df.at[movieIndexID, 'Review'].append()
        df.at[movieIndexID, 'Times Watched'] += 1
        df.to_csv("/Users/ryanbirkedal/Documents/MovieListList.csv", index=False)
        for peeps in searchablePeople:
            if peeps in movie.__dict__['data']:
                movie[peeps] = list(filter(None, movie[peeps]))
                movie[peeps] = list(set(movie[peeps]))
                updatePeopleRatingModule(peeps, movie, peopleDFIn, myRating, movieTitle)
            else:
                continue
    else:

        API = '6ea6649c788824ba87dc8a69da582999'
        imdb = 'https://api.themoviedb.org/3/find/{0}?api_key={1}&language=en-US&external_source=imdb_id'.format(ttimdb,
                                                                                                                 API)
        re = requests.get(url=imdb)
        tmdbExternal = re.json()
        tmdb = tmdbExternal['movie_results'][0]['id']
        streaming = []
        streamingSitesLink = 'https://api.themoviedb.org/3/movie/{0}/watch/providers?api_key={1}'.format(tmdb, API)
        rs = requests.get(url=streamingSitesLink)
        jsonDataStreaming = json.loads(rs.text)
        streamingSites = ['Amazon Prime Video', 'Disney Plus', 'Max', 'Hulu', 'Netflix', 'Paramount Plus',
                          'Peacock Premium', 'Showtime', 'Apple TV Plus',
                          'DIRECTV', 'Hoopla', 'CBS', 'fuboTV', 'MGM Plus', 'AMC+',
                          'USA Network', 'TCM', 'FXNow', 'BroadwayHD', 'Criterion Channel',
                          'Film Movement Plus', 'Starz',
                          'Cinemax Amazon Channel', 'Sling TV Orange and Blue']
        streamList = []
        i = 0
        if 'results' in jsonDataStreaming and 'US' in jsonDataStreaming['results'] and 'flatrate' in \
                jsonDataStreaming['results']['US']:
            while i < len(jsonDataStreaming['results']['US']['flatrate']):
                if jsonDataStreaming['results']['US']['flatrate'][i]['provider_name'] in streamingSites:
                    streamList.append(jsonDataStreaming['results']['US']['flatrate'][i]['provider_name'])
                i += 1
        else:
            streamList.append('No streaming options')
        if len(tmdbExternal['movie_results']) > 0 and tmdbExternal['movie_results'][0]['poster_path'] is not None:
            poster = "https://image.tmdb.org/t/p/w500/" + tmdbExternal['movie_results'][0]['poster_path']
        else:
            poster = "No Poster"

        for criteria in movieCriteria:
            if criteria in movie.__dict__['data']:
                movie[criteria] = list(filter(None, movie[criteria]))
                movie[criteria] = list(set(movie[criteria]))

        commaActors = ",".join(map(str, movie['cast']))
        commaDirectors = ",".join(map(str, movie['director']))
        commaWriters = ",".join(map(str, movie['writer']))
        commaGenres = ",".join(map(str, movie['genres']))
        commaProducers = ",".join(map(str, movie['producer']))
        if 'editor' in movie.__dict__['data']:
            commaEditors = ",".join(map(str, movie['editor']))
        else:
            commaEditors = ""
        if 'production companies' in movie.__dict__['data']:
            commaStudios = ",".join(map(str, movie['production companies']))
        else:
            commaStudios = ""
        if 'cinematographer' in movie.__dict__['data']:
            commaCinematographers = ",".join(map(str, movie['cinematographer']))
        else:
            commaCinematographers = ""
        if 'art direction' in movie.__dict__['data']:
            commaArtDirector = ",".join(map(str, movie['art direction']))
        else:
            commaArtDirector = ""
        if 'production design' in movie.__dict__['data']:
            commaProdDesigner = ",".join(map(str, movie['production design']))
        else:
            commaProdDesigner = ""
        if 'set decoration' in movie.__dict__['data']:
            commaSetDec = ",".join(map(str, movie['set decoration']))
        else:
            commaSetDec = ""
        if 'costume designer' in movie.__dict__['data']:
            commaCostumes = ",".join(map(str, movie['costume designer']))
        else:
            commaCostumes = ""
        if 'make up' in movie.__dict__['data']:
            commaMakeUp = ",".join(map(str, movie['make up']))
        else:
            commaMakeUp = ""
        if 'composer' in movie.__dict__['data']:
            commaComposers = ",".join(map(str, movie['composer']))
        else:
            commaComposers = ""
        if 'box office' in movie.__dict__['data']:
            if 'budget' in movie['box office']:
                budget = movie['box office']['Budget']
            else:
                budget = 'No Data'
            if 'Cumulative Worldwide Gross' in movie['box office']:
                gross = movie['box office']['Cumulative Worldwide Gross']
            else:
                gross = 'No Data'
        else:
            gross = 'No Data'
            budget = 'No Data'
        if 'synopsis' in movie.__dict__['data']:
            synopsis = movie['synopsis']
        else:
            synopsis = ""
        if 'plot' in movie.__dict__['data']:
            plotplot = movie['plot'][0]
        else:
            plotplot = ""

        # gets movie rating in US
        movieRating = "Undefined"
        for rating in movie['certificates']:
            if "United States" in rating and "TV" not in rating:
                movieRating = rating
                movieRating = movieRating.split(":", 1)
        if len(movieRating) > 1 and type(movieRating) == list:
            movieRating = movieRating[1]
        if '::' in movieRating:
            movieRating = movieRating.split('::', 1)
            movieRating = movieRating[0]

        movieInfoIn = pd.DataFrame({'Movie ID': movie['imdbID'], 'Title': movieTitle, 'My Rating': [myRating],
                                    'IMDB Rating': movie['rating'],
                                    'Date Watched': [dateWatched], 'Times Watched': 1, 'Year': movie['year'],
                                    'Studio': commaStudios,
                                    'Runtime': movie['runtimes'], 'Directors': commaDirectors, 'Writers': commaWriters,
                                    'Cinematographers': commaCinematographers,
                                    'Actors': commaActors, 'Genres': commaGenres, 'Composers': commaComposers,
                                    'Editors': commaEditors,
                                    'Producers': commaProducers, 'Costume Director': commaCostumes,
                                    "Make Up Artist": commaMakeUp,
                                    'Art Director': commaArtDirector, 'Production Designer': commaProdDesigner,
                                    'Set Decorator': commaSetDec,
                                    'Plot': plotplot, 'Review': [myReview],
                                    'Synopsis': synopsis, 'Budget': budget,
                                    'Gross': gross, 'Movie Index ID': movieIndexID,
                                    'Rating (Certification)': movieRating, 'poster': poster, 'tmdb': tmdb,
                                    'ttimdb': ttimdb, 'Streaming': [streamList]})
        dfIn = pd.concat([df, movieInfoIn], ignore_index=True)
        dfIn.to_csv("/Users/ryanbirkedal/Documents/MovieListList.csv", index=False)
        for peeps in searchablePeople:
            if peeps in movie.__dict__['data']:
                peopleDFIn = insertPeopleDB(peeps, peopleDFIn, movie, myRating, movieTitle)
                peopleDFIn.to_csv("/Users/ryanbirkedal/Documents/PeopleList.csv", index=False)
            else:
                continue
    return "Added to database", 200


@app.route('/movie/updateForm/<string:ttimdb>', methods=['POST'])
def updateForm(ttimdb):
    newScore = request.form.get('newScore')
    imdb = ttimdb[2:]
    updateMovie = ia.get_movie(imdb)
    updateMovieIndexID = updateMovie['title'][0] + updateMovie['imdbID'] + updateMovie['title'][0]
    movieTitle = updateMovie['title'].replace(',', '')
    df.at[updateMovieIndexID, 'My Rating'].append(newScore)
    df.to_csv("/Users/ryanbirkedal/Documents/MovieListList.csv", index=False)
    searchablePeople = ['cast', 'director', 'writer', 'cinematographer', 'composer', 'genres', 'production companies',
                        'editor', 'producer', 'costume designer', 'make up', 'art direction', 'production design',
                        'set decoration']

    for peeps in searchablePeople:
        if peeps in updateMovie.__dict__['data']:
            updateMovie[peeps] = list(filter(None, updateMovie[peeps]))
            updateMovie[peeps] = list(set(updateMovie[peeps]))
            updatePeopleRatingModule(peeps, updateMovie, peopleDFIn, newScore, movieTitle)
        else:
            continue

    return "Success", 200


@app.route('/movie/removeForm/<string:ttimdb>', methods=['POST'])
def removeForm(ttimdb):
    imdb = ttimdb[2:]
    removeMovie = ia.get_movie(imdb)
    theTitle = removeMovie['title'].replace(',', '')

    searchablePeople = ['cast', 'director', 'writer', 'cinematographer', 'composer', 'genres', 'production companies',
                        'editor', 'producer', 'costume designer', 'make up', 'art direction', 'production design',
                        'set decoration']

    movieIndexID = removeMovie['title'][0] + removeMovie['imdbID'] + removeMovie['title'][0]
    df.drop(index=movieIndexID, inplace=True)
    df.to_csv("/Users/ryanbirkedal/Documents/MovieListList.csv", index=False)
    global peopleDFIn
    for group in searchablePeople:
        if group in removeMovie.__dict__['data']:
            for person in removeMovie[group]:
                if group == 'production companies':
                    castID = person.__dict__['companyID']
                    indexID = str(castID) + group[0] + group[1] + group[2]
                elif group == 'genres':
                    indexID = person
                else:
                    castID = person.__dict__['personID']
                    indexID = str(castID) + group[0] + group[1] + group[2]

                peopleDFIn.set_index('Index ID', inplace=True, drop=False)

                if indexID in peopleDFIn.index:
                    movies = peopleDFIn.at[indexID, 'Movies'].split(',')
                    movies = [m.strip() for m in movies]
                    if theTitle in movies:
                        movieIndex = movies.index(theTitle)
                        movies.remove(theTitle)
                        peopleDFIn.at[indexID, 'Movies'] = ", ".join(map(str, movies))
                        peopleDFIn.at[indexID, 'Movie Count'] -= 1
                        if peopleDFIn.at[indexID, 'Scores'][movieIndex] != 0:
                            peopleDFIn.at[indexID, 'Movies Reviewed'] -= 1
                        scores = peopleDFIn.at[indexID, 'Scores']
                        del scores[movieIndex]
                        if peopleDFIn.at[indexID, 'Movies Reviewed'] == 0:
                            peopleDFIn.at[indexID, 'Average Score'] = 0
                        else:
                            scoreSum = sum(scores)
                            peopleDFIn.at[indexID, 'Average Score'] = scoreSum / float(
                                peopleDFIn.at[indexID, 'Movies Reviewed'])
        else:
            continue
        peopleDFIn.to_csv("/Users/ryanbirkedal/Documents/PeopleList.csv", index=False)
    return "Success", 200


def insertPeopleDB(group, peopleDF, movie, myRating, theTitle):
    for person in movie[group]:
        if group == 'production companies':
            castID = person.__dict__['companyID']
            indexID = str(castID) + group[0] + group[1] + group[2]
            groupName = person
        elif group == 'genres':
            indexID = person
            castID = person
            groupName = person
        else:
            castID = person.__dict__['personID']
            groupName = person['name']
            indexID = str(castID) + group[0] + group[1] + group[2]
        peopleDF.set_index('Index ID', inplace=True, drop=False)
        if indexID not in peopleDF.index:
            if myRating == 0:
                moviesReviewed = 0
            else:
                moviesReviewed = 1
            movieScore = [myRating]
            castInfo = pd.DataFrame([{'Index ID': indexID, 'ID': castID, 'Name': groupName,
                                      'Role': group, 'Movie Count': 1,
                                      'Movies Reviewed': moviesReviewed, 'Movies': theTitle,
                                      'Scores': movieScore, 'Average Score': myRating}])
            peopleDF = pd.concat([peopleDF, castInfo], ignore_index=True)
        else:
            scores = []
            movies = peopleDF.at[indexID, 'Movies'].split(',')
            movies = [m.strip() for m in movies]
            if any(movieSplit == theTitle for movieSplit in movies):
                movieIndex = movies.index(theTitle)
                if peopleDF.at[indexID, 'Scores'][movieIndex] == 0:
                    peopleDF.at[indexID, 'Movies Reviewed'] += 1
                elif peopleDF.at[indexID, 'Scores'][movieIndex] != 0 and myRating == 0:
                    peopleDF.at[indexID, 'Movies Reviewed'] -= 1
                peopleDF.at[indexID, 'Scores'][movieIndex] = myRating
                scores = peopleDF.at[indexID, 'Scores']
                if peopleDF.at[indexID, 'Movies Reviewed'] == 0:
                    peopleDF.at[indexID, 'Average Score'] = 0
                else:
                    scoreSum = sum(scores)
                    peopleDF.at[indexID, 'Average Score'] = scoreSum / float(
                        peopleDF.at[indexID, 'Movies Reviewed'])
            else:
                peopleDF.at[indexID, 'Movie Count'] += 1
                if myRating != 0:
                    peopleDF.at[indexID, 'Movies Reviewed'] += 1
                movies.append(theTitle)
                peopleDF.at[indexID, 'Movies'] = ", ".join(map(str, movies))
                scores = peopleDF.at[indexID, 'Scores']

                scores.append(myRating)
                scoreSum = sum(scores)
                if peopleDF.at[indexID, 'Movies Reviewed'] > 0:
                    peopleDF.at[indexID, 'Average Score'] = scoreSum / float(
                        peopleDF.at[indexID, 'Movies Reviewed'])
                else:
                    peopleDF.at[indexID, 'Average Score'] = 0

    return peopleDF


def updatePeopleRatingModule(group, movie, peopleDF, myRating, theTitle):
    for person in movie[group]:
        if group == 'production companies':
            castID = person.__dict__['companyID']
            indexID = str(castID) + group[0] + group[1] + group[2]
            groupName = person
        elif group == 'genres':
            indexID = person
            castID = person
            groupName = person
        else:
            castID = person.__dict__['personID']
            groupName = person['name']
            indexID = str(castID) + group[0] + group[1] + group[2]

        peopleDF.set_index('Index ID', inplace=True, drop=False)

        if indexID in peopleDF.index:
            movies = peopleDF.at[indexID, 'Movies'].split(',')
            movies = [m.strip() for m in movies]
            if theTitle in movies:
                movieIndex = movies.index(theTitle)
                if peopleDF.at[indexID, 'Scores'][movieIndex] == 0 and myRating != 0:
                    peopleDF.at[indexID, 'Movies Reviewed'] += 1
                elif peopleDF.at[indexID, 'Scores'][movieIndex] != 0 and myRating == 0:
                    peopleDF.at[indexID, 'Movies Reviewed'] -= 1
                peopleDF.at[indexID, 'Scores'][movieIndex] = myRating
                scores = peopleDF.at[indexID, 'Scores']
            else:
                peopleDF.at[indexID, 'Movie Count'] += 1
                if myRating != 0:
                    peopleDF.at[indexID, 'Movies Reviewed'] += 1
                movies.append(theTitle)
                peopleDF.at[indexID, 'Movies'] = ", ".join(map(str, movies))
                scores = peopleDF.at[indexID, 'Scores']
                scores.append(myRating)
            if peopleDF.at[indexID, 'Movies Reviewed'] == 0:
                peopleDF.at[indexID, 'Average Score'] = 0
            else:
                scoreSum = sum(scores)
                peopleDF.at[indexID, 'Average Score'] = scoreSum / float(
                    peopleDF.at[indexID, 'Movies Reviewed'])
        else:
            if myRating == 0:
                moviesReviewed = 0
            else:
                moviesReviewed = 1
            movieScore = [myRating]
            castInfo = pd.DataFrame([{'Index ID': indexID, 'ID': castID, 'Name': groupName,
                                      'Role': group, 'Movie Count': 1,
                                      'Movies Reviewed': moviesReviewed, 'Movies': theTitle,
                                      'Scores': movieScore, 'Average Score': myRating}])
            peopleDF = pd.concat([peopleDF, castInfo], ignore_index=True)
    peopleDF.to_csv("/Users/ryanbirkedal/Documents/PeopleList.csv", index=False)


if __name__ == '__main__':
    app.run(debug=True)
