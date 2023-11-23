from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import *

import pymysql
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
from konlpy.tag import Okt

from gensim.models import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import pearsonr

import joblib

from sqlalchemy import create_engine

from django.apps import apps

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import random

engine = create_engine('mysql+pymysql://admin:pizza715@mml.cu4cw1rqzfei.ap-northeast-2.rds.amazonaws.com/mml?charset=utf8')

mml_user_his = 'SELECT * FROM mml_user_his'
mml_user_his_df = pd.read_sql(mml_user_his, engine)

mml_music_info = 'SELECT * FROM mml_music_info'
mml_music_info_df = pd.read_sql(mml_music_info, engine)        

mml_music_tag = 'SELECT * FROM mml_music_tag'
mml_music_tag_df = pd.read_sql(mml_music_tag, engine)

class music_reco_view(APIView):
    def get(self, request):
        ##user_like_artist
        mml_artist_gen = 'SELECT * FROM mml_artist_gen'
        mml_artist_gen_df = pd.read_sql(mml_artist_gen, engine)

        mml_user_like_artist = 'SELECT * FROM mml_user_like_artist'
        mml_user_like_artist_df = pd.read_sql(mml_user_like_artist, engine)

        mml_user_like_artist_df = mml_user_like_artist_df.rename(columns={'artist_id':'artist'})
        mml_user_like_artist_df.columns

        # 데이터 전처리
        # 사용자가 좋아하는 아티스트 데이터와 아티스트 장르 데이터를 병합하여 좋아하는 아티스트의 장르를 구합니다.
        merged_data = pd.merge(mml_user_like_artist_df, mml_artist_gen_df, on='artist', how='left')

        # 사용자별로 데이터를 그룹화하고 좋아하는 모든 장르, 성별, 연령 그룹을 연결합니다.
        user_genre_df = merged_data.groupby('user_id').agg({
            'genre': lambda x: ' '.join(x.dropna()),
            'gen': 'first',    # 가정: 모든 행에 대해 동일한 값이 존재한다고 가정
            'age_group': 'first'  # 가정: 모든 행에 대해 동일한 값이 존재한다고 가정
        }).reset_index()


        # TF-IDF 벡터 구현
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(user_genre_df['genre'])

        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 사용자 ID를 기반으로 노래를 추천하는 기능 (성별 및 연령대 필터링 적용)
        def recommend_songs(user_id, num_recommendations=3):
            # 사용자 id와 동일한 성별 및 연령대를 가진 사용자들만 필터링
            target_user_data = user_genre_df[user_genre_df['user_id'] == user_id]
            if target_user_data.empty:
                return "사용자 ID를 찾을 수 없습니다."

            target_gen = target_user_data['gen'].iloc[0]
            target_age_group = target_user_data['age_group'].iloc[0]

            filtered_users = user_genre_df[(user_genre_df['gen'] == target_gen) &
                                        (user_genre_df['age_group'] == target_age_group)]

            # 필터링된 사용자들의 인덱스 추출
            filtered_user_indices = filtered_users.index.tolist()

            # 해당 사용자와 필터링된 사용자들의 유사성 점수를 가져옵니다.
            idx = user_genre_df.index[user_genre_df['user_id'] == user_id].tolist()[0]
            sim_scores = list(enumerate(cosine_sim[idx]))

            # 유사성 점수 중에서 필터링된 사용자들만을 대상으로 정렬
            sim_scores = [sim_score for sim_score in sim_scores if sim_score[0] in filtered_user_indices]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # 가장 유사한 사용자들의 점수
            sim_scores = sim_scores[1:num_recommendations+1]

            # 사용자 인덱스
            user_indices = [i[0] for i in sim_scores]

            # 가장 유사한 사용자들 반환
            return user_genre_df['user_id'].iloc[user_indices]

        # 테스트
        test_user_id = request.session.get('username')
        recommended_users = recommend_songs(test_user_id)
        print(recommended_users)

        # 추천 함수를 수정하여 이미 선호하는 아티스트를 제외하고 추천
        def recommend_new_artists(user_id, num_recommendations=10):

            # 원래 사용자의 아티스트 선호도 가져오기
            user_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'] == user_id]['artist'].tolist()

            # 사용자 id에 기반한 추천 받기
            recommended_user_ids = recommend_songs(user_id, num_recommendations).tolist()

            # 추천된 사용자들이 선호하는 아티스트 찾기
            recommended_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'].isin(recommended_user_ids)]['artist']

            # 원래 사용자가 선호하는 아티스트 제외
            new_recommended_artists = recommended_artists[~recommended_artists.isin(user_artists)].unique()

            return new_recommended_artists

        # 이제 추천 함수를 다시 실행하여 테스트 사용자와 유사한 사용자를 찾을 수 있습니다.
        recommended_user_ids = recommend_songs(test_user_id).tolist()

        # 유사한 사용자들이 선호하는 아티스트 찾기
        preferred_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'].isin(recommended_user_ids)]['artist'].unique()

        preferred_artists

        def get_all_songs_for_artists(artist_list):
            songs_dict = {}
            for artist in artist_list:
                # 해당 아티스트의 모든 노래만 필터링
                artist_songs = mml_music_info_df[mml_music_info_df['artist'] == artist]
                # 노래가 존재하는 경우에만 딕셔너리에 추가
                if not artist_songs.empty:
                    songs_dict[artist] = artist_songs['title'].tolist()
            return songs_dict

        # 모든 추천된 아티스트별로 모든 노래를 가져옵니다.
        artist_songs_dict = get_all_songs_for_artists(preferred_artists)

        # 랜덤으로 20곡 선택하기 전에 각 곡에 해당하는 아티스트 정보도 포함시키기
        all_songs_with_artists = []
        for artist, songs in artist_songs_dict.items():
            for song in songs:
                all_songs_with_artists.append((artist, song))

        # 랜덤으로 20곡 선택 (아티스트 정보 포함)
        selected_songs_with_artists = random.sample(all_songs_with_artists, 20)

        # 선택된 노래와 아티스트를 데이터프레임으로 변환
        df_selected_songs_with_artists = pd.DataFrame(selected_songs_with_artists, columns=['artist', 'title'])

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        df_selected_songs_with_artists['title'] = df_selected_songs_with_artists['title'].str.lower()
        df_selected_songs_with_artists['artist'] = df_selected_songs_with_artists['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        user_like_artist_final = pd.merge(
            mml_music_info_df, df_selected_songs_with_artists,
            on=['title', 'artist'],
            how='inner'
        )

        user_like_artist_final = user_like_artist_final[['title', 'artist', 'album_image_url']]


        ##song2vec
        # 모델 로드
        w2v_model = apps.get_app_config('music').model

        processed_lyrics = pd.read_csv('./music/files/processed_lyrics.csv')

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        mml_user_his_df['title'] = mml_user_his_df['title'].str.lower()
        mml_user_his_df['artist'] = mml_user_his_df['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        matched_songs_df = pd.merge(
            mml_music_info_df, mml_user_his_df,
            on=['title', 'artist'],
            how='inner',
            suffixes=('_all_music', '_user_log')
        )

        music_data = matched_songs_df[['user', 'title', 'artist', 'genre_user_log', 'playtime', 'created_at', 'lyrics']]

        music_data = music_data.join(processed_lyrics)

        # Create a reference dataframe from all_music_data with only necessary columns and lowercase transformation for merging
        genre_reference_df = mml_music_info_df[['title', 'artist', 'genre']].copy()
        genre_reference_df['title'] = genre_reference_df['title'].str.lower()
        genre_reference_df['artist'] = genre_reference_df['artist'].str.lower()

        # We create a similar lowercase version of Title and Artist in music_tag for a case-insensitive merge
        music_tag_lowercase = mml_music_tag_df[['title', 'artist', 'tag']].copy()
        music_tag_lowercase['title'] = music_tag_lowercase['title'].str.lower()
        music_tag_lowercase['artist'] = music_tag_lowercase['artist'].str.lower()

        # Merge genre into music_tag using lowercase Title and Artist for matching
        music_tag_data = pd.merge(music_tag_lowercase, genre_reference_df,
                                        left_on=['title', 'artist'], right_on=['title', 'artist'],
                                        how='left')

        # 모든 가사에서 가장 흔한 단어를 추출하고 가중치를 계산하는 함수를 정의합니다.
        def get_top_words_weights(lyrics_list, top_n=20):
            # 모든 가사를 하나의 리스트로 결합합니다.
            all_words = [word for lyrics in lyrics_list for word in lyrics]
            # 가장 흔한 단어와 그 빈도수를 계산합니다.
            top_words = pd.Series(all_words).value_counts().head(top_n)
            # 가중치를 계산합니다: 여기서는 단순화를 위해 빈도수를 그대로 사용하지만,
            # 다른 가중치 할당 방식을 사용할 수도 있습니다.
            weights = top_words / top_words.max()  # 최대 빈도수로 정규화하여 가중치를 계산합니다.
            return weights.to_dict()

        # 사용자별 가장 흔한 단어의 가중치를 계산합니다.
        top_words_weights = get_top_words_weights(music_data['processed_lyrics'])

        # 사용자의 가사 프로필을 만들 때, 가장 흔한 단어에 가중치를 주어 벡터를 계산하는 함수를 수정합니다.
        def create_weighted_lyrics_profile(lyrics_list, w2v_model, top_words_weights):
            lyrics_vectors = []
            for lyrics in lyrics_list:
                # lyrics 벡터의 평균을 계산하기 전에 각 단어에 대한 가중치를 고려합니다.
                weighted_vectors = []
                for word in lyrics:
                    if word in w2v_model.wv:  # 모델의 단어장에 있는 경우에만 처리합니다.
                        weight = top_words_weights.get(word, 1)  # 단어에 대한 가중치를 가져옵니다.
                        weighted_vectors.append(w2v_model.wv[word] * weight)
                if weighted_vectors:  # 가중치가 적용된 벡터의 평균을 계산합니다.
                    lyrics_vectors.append(np.mean(weighted_vectors, axis=0))
            return np.mean(lyrics_vectors, axis=0) if lyrics_vectors else np.zeros(w2v_model.vector_size)

        # 사용자별 프로필 벡터를 생성합니다.
        user_id = request.session.get('username')
        user_lyrics = music_data[music_data['user'] == user_id]['processed_lyrics']
        user_profile_vector = create_weighted_lyrics_profile(user_lyrics, w2v_model, top_words_weights)

        # 특정 사용자 ID에 대한 사용자의 청취 기록을 필터링'02FoMC0v'
        user_specific_log = music_data[music_data['user'] == user_id]

        # 특정 사용자의 장르별 플레이 횟수를 계산
        user_specific_genre_counts = user_specific_log['genre_user_log'].value_counts()

        # 특정 사용자의 상위 5개 장르를 가져옵니다.
        user_specific_top_genres = user_specific_genre_counts.head(5).index.tolist()

        # 사용자 상위 장르와 일치하는 노래에 대해 music_total_with_genre 데이터 프레임 필터링
        user_specific_top_genres_songs_df = music_tag_data[music_tag_data['genre'].isin(user_specific_top_genres)]

        # 태그 데이터를 전처리하는 함수를 정의합니다.
        def preprocess_tags(tag_string):
            # '#' 기호를 기준으로 태그를 분리합니다.
            tags = tag_string.strip().split('#')
            # 빈 문자열을 제거합니다.
            tags = [tag for tag in tags if tag]  # 공백 태그 제거
            return tags

        # 태그 데이터에 전처리 함수를 적용합니다.
        user_specific_top_genres_songs_df['processed_tags'] = user_specific_top_genres_songs_df['tag'].apply(preprocess_tags)

        # 태그를 벡터로 변환하는 함수를 정의합니다.
        def vectorize_tags(tags, w2v_model):
            tag_vectors = []
            for tag in tags:
                # 태그 내의 각 단어에 대해 벡터를 얻고 평균을 계산합니다.
                tag_word_vectors = [w2v_model.wv[word] for word in tag.split() if word in w2v_model.wv]
                if tag_word_vectors:  # 태그가 모델 단어장에 있는 경우에만 평균 벡터를 계산합니다.
                    tag_vectors.append(np.mean(tag_word_vectors, axis=0))
            return np.mean(tag_vectors, axis=0) if tag_vectors else np.zeros(w2v_model.vector_size)

        # 각 태그를 벡터로 변환합니다.
        user_specific_top_genres_songs_df['tag_vector'] = user_specific_top_genres_songs_df['processed_tags'].apply(lambda tags: vectorize_tags(tags, w2v_model))

        # 사용자 프로필 벡터와 모든 태그 벡터 사이의 코사인 유사도를 계산하고 상위 N개의 추천과 함께 유사도를 반환하는 함수
        def recommend_songs_with_similarity(user_profile_vector, tag_vectors, songs_data, top_n=20):
            # 사용자 프로필 벡터를 코사인 유사도 계산을 위해 reshape
            user_vector_reshaped = user_profile_vector.reshape(1, -1)

            # 모든 태그 벡터와의 유사도 계산
            similarity_scores = cosine_similarity(user_vector_reshaped, tag_vectors)[0]

            # 유사도 점수를 기반으로 상위 N개의 인덱스를 가져옵니다
            top_indices = similarity_scores.argsort()[-top_n:][::-1]

            # 상위 N개의 노래 추천 정보와 유사도 점수를 함께 반환
            recommendations_with_scores = songs_data.iloc[top_indices]
            recommendations_with_scores['similarity'] = similarity_scores[top_indices]
            return recommendations_with_scores[['title', 'artist', 'tag', 'similarity']]

        # 모든 태그 벡터를 하나의 배열로 추출합니다.
        tag_vectors_matrix = np.array(list(user_specific_top_genres_songs_df['tag_vector']))

        # 사용자 ID에 대한 노래 추천을 받고 유사도 점수를 포함하여 출력합니다.
        # user_profile_vector_for_similarity = user_profiles[user_id_to_recommend]  # 해당 사용자의 프로필 벡터를 가져옵니다.
        recommendations_with_similarity = recommend_songs_with_similarity(user_profile_vector, tag_vectors_matrix, user_specific_top_genres_songs_df)
        recommendations_with_similarity

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        recommendations_with_similarity['title'] = recommendations_with_similarity['title'].str.lower()
        recommendations_with_similarity['artist'] = recommendations_with_similarity['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        song2vec_final = pd.merge(
            mml_music_info_df, recommendations_with_similarity,
            on=['title', 'artist'],
            how='inner'
        )

        song2vec_final = song2vec_final[['title', 'artist', 'album_image_url']]

        context = (user_like_artist_final, song2vec_final)

        return Response(context, status=status.HTTP_200_OK)

class user_like_artist_view(APIView):
    def get(self, request):

        mml_artist_gen = 'SELECT * FROM mml_artist_gen'
        mml_artist_gen_df = pd.read_sql(mml_artist_gen, engine)

        mml_user_like_artist = 'SELECT * FROM mml_user_like_artist'
        mml_user_like_artist_df = pd.read_sql(mml_user_like_artist, engine)

        mml_user_like_artist_df = mml_user_like_artist_df.rename(columns={'artist_id':'artist'})
        mml_user_like_artist_df.columns

        # 데이터 전처리
        # 사용자가 좋아하는 아티스트 데이터와 아티스트 장르 데이터를 병합하여 좋아하는 아티스트의 장르를 구합니다.
        merged_data = pd.merge(mml_user_like_artist_df, mml_artist_gen_df, on='artist', how='left')

        # 사용자별로 데이터를 그룹화하고 좋아하는 모든 장르, 성별, 연령 그룹을 연결합니다.
        user_genre_df = merged_data.groupby('user_id').agg({
            'genre': lambda x: ' '.join(x.dropna()),
            'gen': 'first',    # 가정: 모든 행에 대해 동일한 값이 존재한다고 가정
            'age_group': 'first'  # 가정: 모든 행에 대해 동일한 값이 존재한다고 가정
        }).reset_index()


        # ITF-IDF 벡터 구현
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(user_genre_df['genre'])

        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 사용자 ID를 기반으로 노래를 추천하는 기능 (성별 및 연령대 필터링 적용)
        def recommend_songs(user_id, num_recommendations=3):
            # 사용자 id와 동일한 성별 및 연령대를 가진 사용자들만 필터링
            target_user_data = user_genre_df[user_genre_df['user_id'] == user_id]
            if target_user_data.empty:
                return "사용자 ID를 찾을 수 없습니다."

            target_gen = target_user_data['gen'].iloc[0]
            target_age_group = target_user_data['age_group'].iloc[0]

            filtered_users = user_genre_df[(user_genre_df['gen'] == target_gen) &
                                        (user_genre_df['age_group'] == target_age_group)]

            # 필터링된 사용자들의 인덱스 추출
            filtered_user_indices = filtered_users.index.tolist()

            # 해당 사용자와 필터링된 사용자들의 유사성 점수를 가져옵니다.
            idx = user_genre_df.index[user_genre_df['user_id'] == user_id].tolist()[0]
            sim_scores = list(enumerate(cosine_sim[idx]))

            # 유사성 점수 중에서 필터링된 사용자들만을 대상으로 정렬
            sim_scores = [sim_score for sim_score in sim_scores if sim_score[0] in filtered_user_indices]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # 가장 유사한 사용자들의 점수
            sim_scores = sim_scores[1:num_recommendations+1]

            # 사용자 인덱스
            user_indices = [i[0] for i in sim_scores]

            # 가장 유사한 사용자들 반환
            return user_genre_df['user_id'].iloc[user_indices]

        # 테스트
        test_user_id = '02FoMC0v'  # 예시 사용자
        recommended_users = recommend_songs(test_user_id)
        print(recommended_users)

        # 추천 함수를 수정하여 이미 선호하는 아티스트를 제외하고 추천
        def recommend_new_artists(user_id, num_recommendations=10):

            # 원래 사용자의 아티스트 선호도 가져오기
            user_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'] == user_id]['artist'].tolist()

            # 사용자 id에 기반한 추천 받기
            recommended_user_ids = recommend_songs(user_id, num_recommendations).tolist()

            # 추천된 사용자들이 선호하는 아티스트 찾기
            recommended_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'].isin(recommended_user_ids)]['artist']

            # 원래 사용자가 선호하는 아티스트 제외
            new_recommended_artists = recommended_artists[~recommended_artists.isin(user_artists)].unique()

            return new_recommended_artists

        # 이제 추천 함수를 다시 실행하여 테스트 사용자와 유사한 사용자를 찾을 수 있습니다.
        recommended_user_ids = recommend_songs(test_user_id).tolist()

        # 유사한 사용자들이 선호하는 아티스트 찾기
        preferred_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'].isin(recommended_user_ids)]['artist'].unique()

        preferred_artists

        def get_all_songs_for_artists(artist_list):
            songs_dict = {}
            for artist in artist_list:
                # 해당 아티스트의 모든 노래만 필터링
                artist_songs = mml_music_info_df[mml_music_info_df['artist'] == artist]
                # 노래가 존재하는 경우에만 딕셔너리에 추가
                if not artist_songs.empty:
                    songs_dict[artist] = artist_songs['title'].tolist()
            return songs_dict

        # 모든 추천된 아티스트별로 모든 노래를 가져옵니다.
        artist_songs_dict = get_all_songs_for_artists(preferred_artists)

        # 랜덤으로 20곡 선택하기 전에 각 곡에 해당하는 아티스트 정보도 포함시키기
        all_songs_with_artists = []
        for artist, songs in artist_songs_dict.items():
            for song in songs:
                all_songs_with_artists.append((artist, song))

        # 랜덤으로 20곡 선택 (아티스트 정보 포함)
        selected_songs_with_artists = random.sample(all_songs_with_artists, 20)

        # 선택된 노래와 아티스트를 데이터프레임으로 변환
        df_selected_songs_with_artists = pd.DataFrame(selected_songs_with_artists, columns=['artist', 'title'])

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        df_selected_songs_with_artists['title'] = df_selected_songs_with_artists['title'].str.lower()
        df_selected_songs_with_artists['artist'] = df_selected_songs_with_artists['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        user_like_artist_final = pd.merge(
            mml_music_info_df, df_selected_songs_with_artists,
            on=['title', 'artist'],
            how='inner'
        )

        user_like_artist_final = user_like_artist_final[['title', 'artist', 'album_image_url']]

        return Response(user_like_artist_final.head(20), status=status.HTTP_200_OK)


class song2vec_view(APIView):
    def get(self, request):
    
        # 모델 로드
        w2v_model = apps.get_app_config('music').model

        processed_lyrics = pd.read_csv('./music/files/processed_lyrics.csv')

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        mml_user_his_df['title'] = mml_user_his_df['title'].str.lower()
        mml_user_his_df['artist'] = mml_user_his_df['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        matched_songs_df = pd.merge(
            mml_music_info_df, mml_user_his_df,
            on=['title', 'artist'],
            how='inner',
            suffixes=('_all_music', '_user_log')
        )

        # Since the user wants to add lyrics to the user music log based on the title and artist match,
        # we will merge the 'user_music_log_with_genres' with 'new_all_music_data' on 'Title' and 'Artist'
        # to add the 'Lyrics' column to the user music log dataframe.

        # We will use the 'matched_songs_df' which already has the matched songs to select the needed columns
        # We will create a new dataframe with the 'Lyrics' column included

        # Selecting only the necessary columns to include in the final merged dataframe
        music_data = matched_songs_df[['user', 'title', 'artist', 'genre_user_log', 'playtime', 'created_at', 'lyrics']]

        music_data = music_data.join(processed_lyrics)

        # I made a mistake in the merge and selection of columns process. Let's correct that.

        # Since we are interested in adding 'Genre' from 'all_music_data' to 'music_tag',
        # first we will create a smaller dataframe of 'all_music_data' with only 'Title', 'Artist', and 'Genre' columns
        # with lowercase conversion for merging.

        # Create a reference dataframe from all_music_data with only necessary columns and lowercase transformation for merging
        genre_reference_df = mml_music_info_df[['title', 'artist', 'genre']].copy()
        genre_reference_df['title'] = genre_reference_df['title'].str.lower()
        genre_reference_df['artist'] = genre_reference_df['artist'].str.lower()

        # We create a similar lowercase version of Title and Artist in music_tag for a case-insensitive merge
        music_tag_lowercase = mml_music_tag_df[['title', 'artist', 'tag']].copy()
        music_tag_lowercase['title'] = music_tag_lowercase['title'].str.lower()
        music_tag_lowercase['artist'] = music_tag_lowercase['artist'].str.lower()

        # Merge genre into music_tag using lowercase Title and Artist for matching
        music_tag_data = pd.merge(music_tag_lowercase, genre_reference_df,
                                        left_on=['title', 'artist'], right_on=['title', 'artist'],
                                        how='left')

        # 모든 가사에서 가장 흔한 단어를 추출하고 가중치를 계산하는 함수를 정의합니다.
        def get_top_words_weights(lyrics_list, top_n=20):
            # 모든 가사를 하나의 리스트로 결합합니다.
            all_words = [word for lyrics in lyrics_list for word in lyrics]
            # 가장 흔한 단어와 그 빈도수를 계산합니다.
            top_words = pd.Series(all_words).value_counts().head(top_n)
            # 가중치를 계산합니다: 여기서는 단순화를 위해 빈도수를 그대로 사용하지만,
            # 다른 가중치 할당 방식을 사용할 수도 있습니다.
            weights = top_words / top_words.max()  # 최대 빈도수로 정규화하여 가중치를 계산합니다.
            return weights.to_dict()

        # 사용자별 가장 흔한 단어의 가중치를 계산합니다.
        top_words_weights = get_top_words_weights(music_data['processed_lyrics'])

        # 사용자의 가사 프로필을 만들 때, 가장 흔한 단어에 가중치를 주어 벡터를 계산하는 함수를 수정합니다.
        def create_weighted_lyrics_profile(lyrics_list, w2v_model, top_words_weights):
            lyrics_vectors = []
            for lyrics in lyrics_list:
                # lyrics 벡터의 평균을 계산하기 전에 각 단어에 대한 가중치를 고려합니다.
                weighted_vectors = []
                for word in lyrics:
                    if word in w2v_model.wv:  # 모델의 단어장에 있는 경우에만 처리합니다.
                        weight = top_words_weights.get(word, 1)  # 단어에 대한 가중치를 가져옵니다.
                        weighted_vectors.append(w2v_model.wv[word] * weight)
                if weighted_vectors:  # 가중치가 적용된 벡터의 평균을 계산합니다.
                    lyrics_vectors.append(np.mean(weighted_vectors, axis=0))
            return np.mean(lyrics_vectors, axis=0) if lyrics_vectors else np.zeros(w2v_model.vector_size)

        # 사용자별 프로필 벡터를 생성합니다.
        user_id = request.session.get('username')
        user_lyrics = music_data[music_data['user'] == user_id]['processed_lyrics']
        user_profile_vector = create_weighted_lyrics_profile(user_lyrics, w2v_model, top_words_weights)

        # 특정 사용자 ID에 대한 사용자의 청취 기록을 필터링'02FoMC0v'
        user_specific_log = music_data[music_data['user'] == user_id]

        # 특정 사용자의 장르별 플레이 횟수를 계산
        user_specific_genre_counts = user_specific_log['genre_user_log'].value_counts()

        # 특정 사용자의 상위 3개 장르를 가져옵니다.
        user_specific_top_genres = user_specific_genre_counts.head(5).index.tolist()

        # 사용자 상위 장르와 일치하는 노래에 대해 music_total_with_genre 데이터 프레임 필터링
        user_specific_top_genres_songs_df = music_tag_data[music_tag_data['genre'].isin(user_specific_top_genres)]

        # 태그 데이터를 전처리하는 함수를 정의합니다.
        def preprocess_tags(tag_string):
            # '#' 기호를 기준으로 태그를 분리합니다.
            tags = tag_string.strip().split('#')
            # 빈 문자열을 제거합니다.
            tags = [tag for tag in tags if tag]  # 공백 태그 제거
            return tags

        # 태그 데이터에 전처리 함수를 적용합니다.
        user_specific_top_genres_songs_df['processed_tags'] = user_specific_top_genres_songs_df['tag'].apply(preprocess_tags)

        # 태그를 벡터로 변환하는 함수를 정의합니다.
        def vectorize_tags(tags, w2v_model):
            tag_vectors = []
            for tag in tags:
                # 태그 내의 각 단어에 대해 벡터를 얻고 평균을 계산합니다.
                tag_word_vectors = [w2v_model.wv[word] for word in tag.split() if word in w2v_model.wv]
                if tag_word_vectors:  # 태그가 모델 단어장에 있는 경우에만 평균 벡터를 계산합니다.
                    tag_vectors.append(np.mean(tag_word_vectors, axis=0))
            return np.mean(tag_vectors, axis=0) if tag_vectors else np.zeros(w2v_model.vector_size)

        # 각 태그를 벡터로 변환합니다.
        user_specific_top_genres_songs_df['tag_vector'] = user_specific_top_genres_songs_df['processed_tags'].apply(lambda tags: vectorize_tags(tags, w2v_model))

        # 사용자 프로필 벡터와 모든 태그 벡터 사이의 코사인 유사도를 계산하고 상위 N개의 추천과 함께 유사도를 반환하는 함수
        def recommend_songs_with_similarity(user_profile_vector, tag_vectors, songs_data, top_n=20):
            # 사용자 프로필 벡터를 코사인 유사도 계산을 위해 reshape
            user_vector_reshaped = user_profile_vector.reshape(1, -1)

            # 모든 태그 벡터와의 유사도 계산
            similarity_scores = cosine_similarity(user_vector_reshaped, tag_vectors)[0]

            # 유사도 점수를 기반으로 상위 N개의 인덱스를 가져옵니다
            top_indices = similarity_scores.argsort()[-top_n:][::-1]

            # 상위 N개의 노래 추천 정보와 유사도 점수를 함께 반환
            recommendations_with_scores = songs_data.iloc[top_indices]
            recommendations_with_scores['similarity'] = similarity_scores[top_indices]
            return recommendations_with_scores[['title', 'artist', 'tag', 'similarity']]

        # 모든 태그 벡터를 하나의 배열로 추출합니다.
        tag_vectors_matrix = np.array(list(user_specific_top_genres_songs_df['tag_vector']))

        # 사용자 ID에 대한 노래 추천을 받고 유사도 점수를 포함하여 출력합니다.
        # user_profile_vector_for_similarity = user_profiles[user_id_to_recommend]  # 해당 사용자의 프로필 벡터를 가져옵니다.
        recommendations_with_similarity = recommend_songs_with_similarity(user_profile_vector, tag_vectors_matrix, user_specific_top_genres_songs_df)
        recommendations_with_similarity

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        recommendations_with_similarity['title'] = recommendations_with_similarity['title'].str.lower()
        recommendations_with_similarity['artist'] = recommendations_with_similarity['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        song2vec_final = pd.merge(
            mml_music_info_df, recommendations_with_similarity,
            on=['title', 'artist'],
            how='inner'
        )

        song2vec_final = song2vec_final[['title', 'artist', 'album_image_url']]

        return Response(song2vec_final.head(20), status=status.HTTP_200_OK)
    
class tag_song2vec_view(APIView):
    def get(self, request):

        # 모델 로드
        w2v_model = apps.get_app_config('music').model

        processed_lyrics = pd.read_csv('./music/files/processed_lyrics.csv')

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        mml_user_his_df['title'] = mml_user_his_df['title'].str.lower()
        mml_user_his_df['artist'] = mml_user_his_df['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        matched_songs_df = pd.merge(
            mml_music_info_df, mml_user_his_df,
            on=['title', 'artist'],
            how='inner',
            suffixes=('_all_music', '_user_log')
        )

        music_data = matched_songs_df[['user', 'title', 'artist', 'genre_user_log', 'playtime', 'created_at', 'lyrics']]

        music_data = music_data.join(processed_lyrics)

        # Input sentence from the user
        input_sentence = request.data.get('sentence', '')

        # Tokenizing the sentence
        tokens = word_tokenize(input_sentence)

        # Removing stopwords (commonly used words that are not useful for keyword extraction)
        # Note: For Korean, a custom list of stopwords might be needed as nltk's default is for English.
        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

        # Checking the unique values in the 'Title' column to understand its format
        unique_titles = mml_music_tag_df['title'].unique()

        # Processing the 'Tag' column: Splitting the tags into separate entries
        # We create a new DataFrame with each tag as a separate entry
        tag_data_expanded = mml_music_tag_df.drop('tag', axis=1).join(
            mml_music_tag_df['tag'].str.split('#', expand=True).stack().reset_index(level=1, drop=True).rename('tag')
        )

        # Converting all titles to string
        mml_music_tag_df['title'] = mml_music_tag_df['title'].astype(str)

        # Now, we will process the 'Tag' data for text similarity analysis.
        # First, we remove empty tags and convert tags to lower case for uniformity
        tag_data_expanded = tag_data_expanded[tag_data_expanded['tag'].str.strip().ne('')]
        tag_data_expanded['tag'] = tag_data_expanded['tag'].str.lower()

        # Creating a list of unique tags for TF-IDF
        unique_tags = tag_data_expanded['tag'].unique()

        # Initializing the TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fitting the vectorizer to the unique tags
        tfidf_matrix = tfidf_vectorizer.fit_transform(unique_tags)

        # Updating the input sentence to include multiple keywords
        input_keywords = filtered_tokens
        # Vectorizing each keyword separately
        input_vectors = [tfidf_vectorizer.transform([keyword]) for keyword in input_keywords]

        # Calculating cosine similarity for each keyword with all tags
        cosine_similarities_keywords = [cosine_similarity(input_vector, tfidf_matrix).flatten() for input_vector in input_vectors]

        # Finding the most similar tag for each keyword
        most_similar_tags = [unique_tags[np.argmax(cosine_similarities)] for cosine_similarities in cosine_similarities_keywords]
        similarity_scores = [np.max(cosine_similarities) for cosine_similarities in cosine_similarities_keywords]

        # Creating a dictionary to display results for each keyword
        similarity_results = dict(zip(input_keywords, zip(most_similar_tags, similarity_scores)))
        similarity_results

        # 이전 단계에서 계산한 가장 유사한 태그 사용
        # 예를 들어, 'similarity_results' 딕셔너리에서 태그 추출
        most_similar_tags = [similarity_results[keyword][0] for keyword in input_keywords]

        # 해당 태그를 포함하는 음악 리스트 추출
        music_list = [df for df in music_list if df is not None and not df.empty]
        for tag in most_similar_tags:
            # 해당 태그를 포함하는 모든 음악 찾기
            music_with_tag = mml_music_tag_df[mml_music_tag_df['tag'].str.contains(tag)]
            music_list.append(music_with_tag)

        # 모든 결과를 하나의 DataFrame으로 결합
        similar_mml_music_tag_df = pd.concat(music_list).drop_duplicates().reset_index(drop=True)

        genre_reference_df = mml_music_info_df[['title', 'artist', 'genre']].copy()
        genre_reference_df['title'] = genre_reference_df['title'].str.lower()
        genre_reference_df['artist'] = genre_reference_df['artist'].str.lower()

        # We create a similar lowercase version of Title and Artist in music_tag for a case-insensitive merge
        music_tag_lowercase = similar_mml_music_tag_df[['title', 'artist', 'tag']].copy()
        music_tag_lowercase['title'] = music_tag_lowercase['title'].str.lower()
        music_tag_lowercase['artist'] = music_tag_lowercase['artist'].str.lower()

        # Merge genre into music_tag using lowercase Title and Artist for matching
        music_tag_data = pd.merge(music_tag_lowercase, genre_reference_df,
                                        left_on=['title', 'artist'], right_on=['title', 'artist'],
                                        how='left')

        # 모든 가사에서 가장 흔한 단어를 추출하고 가중치를 계산하는 함수를 정의합니다.
        def get_top_words_weights(lyrics_list, top_n=20):
            # 모든 가사를 하나의 리스트로 결합합니다.
            all_words = [word for lyrics in lyrics_list for word in lyrics]
            # 가장 흔한 단어와 그 빈도수를 계산합니다.
            top_words = pd.Series(all_words).value_counts().head(top_n)
            # 가중치를 계산합니다: 여기서는 단순화를 위해 빈도수를 그대로 사용하지만,
            # 다른 가중치 할당 방식을 사용할 수도 있습니다.
            weights = top_words / top_words.max()  # 최대 빈도수로 정규화하여 가중치를 계산합니다.
            return weights.to_dict()

        # 사용자별 가장 흔한 단어의 가중치를 계산합니다.
        top_words_weights = get_top_words_weights(music_data['processed_lyrics'])

        # 사용자의 가사 프로필을 만들 때, 가장 흔한 단어에 가중치를 주어 벡터를 계산하는 함수를 수정합니다.
        def create_weighted_lyrics_profile(lyrics_list, w2v_model, top_words_weights):
            lyrics_vectors = []
            for lyrics in lyrics_list:
                # lyrics 벡터의 평균을 계산하기 전에 각 단어에 대한 가중치를 고려합니다.
                weighted_vectors = []
                for word in lyrics:
                    if word in w2v_model.wv:  # 모델의 단어장에 있는 경우에만 처리합니다.
                        weight = top_words_weights.get(word, 1)  # 단어에 대한 가중치를 가져옵니다.
                        weighted_vectors.append(w2v_model.wv[word] * weight)
                if weighted_vectors:  # 가중치가 적용된 벡터의 평균을 계산합니다.
                    lyrics_vectors.append(np.mean(weighted_vectors, axis=0))
            return np.mean(lyrics_vectors, axis=0) if lyrics_vectors else np.zeros(w2v_model.vector_size)

        # 사용자별 프로필 벡터를 생성합니다.
        user_id = request.session.get('username')
        user_lyrics = music_data[music_data['user'] == user_id]['processed_lyrics']
        user_profile_vector = create_weighted_lyrics_profile(user_lyrics, w2v_model, top_words_weights)

        # 특정 사용자 ID에 대한 사용자의 청취 기록을 필터링'02FoMC0v'
        user_specific_log = music_data[music_data['user'] == user_id]

        # 특정 사용자의 장르별 플레이 횟수를 계산
        user_specific_genre_counts = user_specific_log['genre_user_log'].value_counts()

        # 특정 사용자의 상위 3개 장르를 가져옵니다.
        user_specific_top_genres = user_specific_genre_counts.head(5).index.tolist()

        # 사용자 상위 장르와 일치하는 노래에 대해 music_total_with_genre 데이터 프레임 필터링
        user_specific_top_genres_songs_df = music_tag_data[music_tag_data['genre'].isin(user_specific_top_genres)]

        # 태그 데이터를 전처리하는 함수를 정의합니다.
        def preprocess_tags(tag_string):
            # '#' 기호를 기준으로 태그를 분리합니다.
            tags = tag_string.strip().split('#')
            # 빈 문자열을 제거합니다.
            tags = [tag for tag in tags if tag]  # 공백 태그 제거
            return tags

        # 태그 데이터에 전처리 함수를 적용합니다.
        user_specific_top_genres_songs_df['processed_tags'] = user_specific_top_genres_songs_df['tag'].apply(preprocess_tags)

        # 태그를 벡터로 변환하는 함수를 정의합니다.
        def vectorize_tags(tags, w2v_model):
            tag_vectors = []
            for tag in tags:
                # 태그 내의 각 단어에 대해 벡터를 얻고 평균을 계산합니다.
                tag_word_vectors = [w2v_model.wv[word] for word in tag.split() if word in w2v_model.wv]
                if tag_word_vectors:  # 태그가 모델 단어장에 있는 경우에만 평균 벡터를 계산합니다.
                    tag_vectors.append(np.mean(tag_word_vectors, axis=0))
            return np.mean(tag_vectors, axis=0) if tag_vectors else np.zeros(w2v_model.vector_size)

        # 각 태그를 벡터로 변환합니다.
        user_specific_top_genres_songs_df['tag_vector'] = user_specific_top_genres_songs_df['processed_tags'].apply(lambda tags: vectorize_tags(tags, w2v_model))

        # 사용자 프로필 벡터와 모든 태그 벡터 사이의 코사인 유사도를 계산하고 상위 N개의 추천과 함께 유사도를 반환하는 함수
        def recommend_songs_with_similarity(user_profile_vector, tag_vectors, songs_data, top_n=20):
            # 사용자 프로필 벡터를 코사인 유사도 계산을 위해 reshape
            user_vector_reshaped = user_profile_vector.reshape(1, -1)

            # 모든 태그 벡터와의 유사도 계산
            similarity_scores = cosine_similarity(user_vector_reshaped, tag_vectors)[0]

            # 유사도 점수를 기반으로 상위 N개의 인덱스를 가져옵니다
            top_indices = similarity_scores.argsort()[-top_n:][::-1]

            # 상위 N개의 노래 추천 정보와 유사도 점수를 함께 반환
            recommendations_with_scores = songs_data.iloc[top_indices]
            recommendations_with_scores['similarity'] = similarity_scores[top_indices]
            return recommendations_with_scores[['title', 'artist', 'tag', 'similarity']]

        # 모든 태그 벡터를 하나의 배열로 추출합니다.
        tag_vectors_matrix = np.array(list(user_specific_top_genres_songs_df['tag_vector']))

        # 사용자 ID에 대한 노래 추천을 받고 유사도 점수를 포함하여 출력합니다.
        # user_profile_vector_for_similarity = user_profiles[user_id_to_recommend]  # 해당 사용자의 프로필 벡터를 가져옵니다.
        recommendations_with_similarity = recommend_songs_with_similarity(user_profile_vector, tag_vectors_matrix, user_specific_top_genres_songs_df)
        recommendations_with_similarity

        # Normalize the 'Title' and 'Artist' columns in both dataframes for case-insensitive comparison
        mml_music_info_df['title'] = mml_music_info_df['title'].str.lower()
        mml_music_info_df['artist'] = mml_music_info_df['artist'].str.lower()
        recommendations_with_similarity['title'] = recommendations_with_similarity['title'].str.lower()
        recommendations_with_similarity['artist'] = recommendations_with_similarity['artist'].str.lower()

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        tag_song2vec_final = pd.merge(
            mml_music_info_df, recommendations_with_similarity,
            on=['title', 'artist'],
            how='inner'
        )

        tag_song2vec_final = tag_song2vec_final[['title', 'artist', 'album_image_url']]

        return Response(tag_song2vec_final.head(20), status=status.HTTP_200_OK)