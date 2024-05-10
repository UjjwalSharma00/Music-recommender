import pandas as pd
import numpy as np

import os
os.chdir ('Z:\\OneDrive\\Desktop\\Projects\\Music Recommendation System\\')
import Recommenders as Recommenders

# Load the Data - triplets_file

song_df_1 = pd.read_csv('triplets_file.csv')
print (song_df_1.shape)
print (song_df_1.head())

# Load the Data - song_data

song_df_2 = pd.read_csv('song_data.csv')
print (song_df_2.shape)
display  (song_df_2.head())


# Combine two data frames and create one data Frame
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')
display (song_df.shape)
display (song_df.head())


# length of each data Frame 
print(len(song_df_1), len(song_df_2))

# Length of the consolidated data frame 
len(song_df)

# Select only 50000 records to create a model to improve performance
song_df = song_df.head(50000)
song_df.shape

# Combining title and artist name
song_df['song'] = song_df['title']+' - '+song_df['artist_name']
song_df.head()


# Cumulative sum of listen count of the songs
# Group by based on Song

song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
song_grouped.head()


# Cumulative sum of listen count of the songs
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
song_grouped.head()


# Total number of records
grouped_sum = song_grouped['listen_count'].sum()
grouped_sum

# Display the percentage to identify most popular song
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum ) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])

# Popularity Recommendation Engine  -------------------------------

# Import Popularity Recommender Model
pr = Recommenders.popularity_recommender_py()
pr.create(song_df, 'user_id', 'song')

# Display the top 10 popular songs- User 5
pr.recommend(song_df['user_id'][5])

# Display the top 10 popular songs- User 100
# Popularity rating same for all users
pr.recommend(song_df['user_id'][100])


# Item Similarity Recommendation  --------------------------------------

# Import Item Similarity Model 

ir = Recommenders.item_similarity_recommender_py()
ir.create(song_df, 'user_id', 'song')

# Item Similarity for User -5
user_items = ir.get_user_items(song_df['user_id'][5])  # display user songs history
for user_item in user_items:                   
    print(user_item)

# Item Similarity for User -55

user_items = ir.get_user_items(song_df['user_id'][55])
# display user songs history
for user_item in user_items:
    print(user_item)


# Song recommendation for that user-5
ir.recommend(song_df['user_id'][5])


# Song recommendation for that user-25
ir.recommend(song_df['user_id'][25])


# Recommendation based on Song Name ----------------------

# Based on selected song provide recommendation 
ir.get_similar_items(['Oliver James - Fleet Foxes', 'The End - Pearl Jam'])

# Recommendation for another song
ir.get_similar_items(['Use Somebody - Kings Of Leon'])