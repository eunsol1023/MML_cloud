# data_loader.py
import pandas as pd
from sqlalchemy import create_engine

class artist_DataLoader:
    def __init__(self, engine):
        self.engine = engine

    def artist_load_data(self):

        mml_music_info = 'SELECT * FROM mml_music_info'
        mml_music_info_df = pd.read_sql(mml_music_info, self.engine)
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()        

        mml_artist_gen = 'SELECT * FROM mml_artist_gen'
        mml_artist_gen_df = pd.read_sql(mml_artist_gen, self.engine)
        mml_artist_gen_df['artist'] = mml_artist_gen_df['artist'].str.lower() 

        mml_user_like_artist = 'SELECT * FROM mml_user_like_artist'
        mml_user_like_artist_df = pd.read_sql(mml_user_like_artist, self.engine)
        mml_user_like_artist_df['artist_id'] = mml_user_like_artist_df['artist_id'].str.lower()
        mml_user_like_artist_df = mml_user_like_artist_df.rename(columns={'artist_id':'artist'})

        return mml_music_info_df, mml_artist_gen_df, mml_user_like_artist_df