# data_loader.py
import pandas as pd
from sqlalchemy import create_engine

class song2vec_DataLoader:
    def __init__(self, engine):
        self.engine = engine

    def song2vec_load_data(self):
        mml_user_his = 'SELECT * FROM mml_user_his'
        mml_user_his_df = pd.read_sql(mml_user_his, self.engine)
        mml_user_his_df['title'] = mml_user_his_df['title'].str.lower()
        mml_user_his_df['artist'] = mml_user_his_df['artist'].str.lower()

        mml_music_info = 'SELECT * FROM mml_music_info'
        mml_music_info_df = pd.read_sql(mml_music_info, self.engine)
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()        

        mml_music_tag = 'SELECT * FROM mml_music_tag'
        mml_music_tag_df = pd.read_sql(mml_music_tag, self.engine)
        mml_music_tag_df['title'] = mml_music_tag_df['title'].str.lower()
        mml_music_tag_df['artist'] = mml_music_tag_df['artist'].str.lower()


        matched_songs_df = pd.merge(
        mml_music_info_df, mml_user_his_df,
        on=['title', 'artist'],
        how='inner',
        suffixes=('_all_music', '_user_log')
            )

        processed_lyrics = pd.read_csv('./music/files/processed_lyrics.csv')

        # Selecting only the necessary columns to include in the final merged dataframe
        music_data = matched_songs_df[['user', 'title', 'artist', 'genre_user_log', 'playtime', 'created_at', 'lyrics']]

        music_data = music_data.join(processed_lyrics)

        # Create a reference dataframe from all_music_data with only necessary columns and lowercase transformation for merging
        genre_reference_df = mml_music_info_df[['title', 'artist', 'genre']].copy()

        # We create a similar lowercase version of Title and Artist in music_tag for a case-insensitive merge
        music_tag_lowercase = mml_music_tag_df[['title', 'artist', 'tag']].copy()

        # Merge genre into music_tag using lowercase Title and Artist for matching
        music_tag_data = pd.merge(music_tag_lowercase, genre_reference_df,
                                        left_on=['title', 'artist'], right_on=['title', 'artist'],
                                        how='left')
        
        return mml_user_his_df, mml_music_info_df, mml_music_tag_df, music_data, music_tag_data