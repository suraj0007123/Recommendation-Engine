### Importing Libraries 
import pandas as pd
import numpy as np

##### Loading the game dataset for recommendation analysis
gamedata=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\recommendation engine\Datasets_Recommendation Engine\game.csv",encoding='utf-8')

gamedata.shape ## To know the shape of the dataset

gamedata.columns ### To know the columns names of dataset

gamedata.rating
gamedata.userId
gamedata.game

gamedata.head(20)

### Identifying the duplicates 
gamedata.duplicated().sum()

##importing the tfidfvectorizer term frequenecy inverse document apply when you have text data 
from sklearn.feature_extraction.text import TfidfVectorizer 

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words='english') # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
gamedata['game'].isnull().sum()

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix =tfidf.fit_transform(gamedata['game'])  #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #5000, 3068

##now doing pair wise similiraty score using cosine matrix
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number
game_index = pd.Series(gamedata.index, index = gamedata['game']).drop_duplicates()
userId = game_index['Mortal Kombat']
userId
userId = game_index['Need for Speed: Most Wanted U']
userId
userId = game_index['Need for Speed: Carbon']
userId

def get_recommendations(game, topN):
    # topN = 10 games 
   
    userId = game_index[game]
    #apirwise similarity getting for all the UserId's
    cosine_scores = list(enumerate(cosine_sim_matrix[userId]))
    
    #sorting the cosine similarity based on scores
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    
    #getting the scores of top n similar Games
    cosine_scores_N = cosine_scores[0: topN+1]
    
    #getting the movie index
    userIdx = [i[0] for i in cosine_scores_N]
    gamedata_scores = [i[1] for i in cosine_scores_N]
    
    #similar movies and scores
    gamedata_similar_show = pd.DataFrame(columns=['game', 'Score'])
    gamedata_similar_show['game'] = gamedata.loc[userIdx, 'game']
    gamedata_similar_show['Score'] = gamedata_scores
    gamedata_similar_show.reset_index(inplace=True)
    print(gamedata_similar_show)
    
#entering name of Top-Games  to remcommend    
get_recommendations('Need for Speed: Hot Pursuit 2', topN = 10)
game_index['Need for Speed: Hot Pursuit 2']

get_recommendations('Super Paper Mario', topN = 10)
game_index['Super Paper Mario']

get_recommendations('Donut County', topN = 10)
game_index['Donut County']
