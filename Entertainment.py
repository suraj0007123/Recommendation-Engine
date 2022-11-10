### Importing Libraries 
import pandas as pd
import numpy as np

####### Loading the Entertainment dataset for recommendation analysis
entertainment=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\recommendation engine\Datasets_Recommendation Engine\Entertainment.csv")

entertainment.shape ### To know the shape of the dataset

entertainment.columns ## To know the column names of the dataset

entertainment.Id

entertainment.Titles
entertainment.Category ## TO know the category of all the columns
entertainment.Reviews

entertainment.dtypes
entertainment.head(10)

### Identiffying the duplicates ###
entertainment.duplicated().sum()


##importing the tfidfvectorizer term frequenecy inverse document apply when you have text data
from sklearn.feature_extraction.text import TfidfVectorizer

#creating tfidf vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words='english')

entertainment['Category'].isnull().sum() ## Checking is no nan values is in the  category column



## Now let as preaparing tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(entertainment.Category)
tfidf_matrix.shape  ### 51 , 34  This is the shape tfidf_matrix

##now doing pair wise similiraty score using cosine matrix
from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

##creating and mapping of entertainment name to index number for checking the id number 
entertainment_index = pd.Series(entertainment.index, index = entertainment['Titles']).drop_duplicates()
entertainment_Id = entertainment_index["Casino (1995)"]
entertainment_Id

def get_recommendations(Name, topN):
    ##taking the Top 10 movies
    entertainment_Id = entertainment_index[Name]
    
    cosine_scores = list(enumerate(cosine_sim_matrix[entertainment_Id]))#apirwise similarity getting the score
    
    #sorting the cosine similarity based on scores by using the lambda function as a key
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1],reverse=True)
    
    #getting the scores of all top n similar movies
    cosine_scores_N = cosine_scores[0: topN+1]
    
    #getting the movie index
    enter_Id = [i[0] for i in cosine_scores_N]
    enter_scores = [i[1] for i in cosine_scores_N]
    
    #similar movies and scores
    entertainment_similar_show = pd.DataFrame(columns=["Titles", "Scores"])
    
    entetainment_similar_show['Titles'] = entertainment.loc[entertainment_Id, Titles]
    
    entertainment_similar_show['Score'] = entertainment_scores
    
    entertainment_similar_show.reset_index(inplace=True)
    
    print(entertainment_similar_show)
    

#ntering name of entertaiment to remcommend
get_recommendations('Sabrina (1995)',topN=10)

entertainment_index['Jumanji (1995)']
