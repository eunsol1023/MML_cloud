from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import *

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from konlpy.tag import Okt


from scipy.stats import pearsonr

from sqlalchemy import create_engine

from django.apps import apps

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from song2vec_data_loader import song2vec_DataLoader

engine = create_engine('mysql+pymysql://admin:pizza715@mml.cu4cw1rqzfei.ap-northeast-2.rds.amazonaws.com/mml?charset=utf8')

# DataLoader 인스턴스 생성
song2vec_data_loader = song2vec_DataLoader(engine)

mml_user_his_df, mml_music_info_df, mml_music_tag_df, music_data, music_tag_data = song2vec_data_loader.song2vec_load_data()

user_id = '08XxwFym'

class tag_song2vec_view(APIView):
    def get(self, request):
        print('4번')

        # 모델 로드
        w2v_model = apps.get_app_config('music').model

        processed_lyrics = pd.read_csv('./music/files/processed_lyrics.csv')

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
        input_sentence = request.query_params.get('input_sentence', None)
        print("input_sentence의 값 : ", input_sentence)
        print("input_sentence의 타입:", type(input_sentence))
        if not input_sentence:
            return Response({'error': 'input_sentence가 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)

        # Tokenizing the sentence
        tokens = word_tokenize(input_sentence)

        # Removing stopwords (commonly used words that are not useful for keyword extraction)
        # Note: For Korean, a custom list of stopwords might be needed as nltk's default is for English.
        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

        filtered_tokens

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
        music_list = []
        for tag in most_similar_tags:
            # 해당 태그를 포함하는 모든 음악 찾기
            music_with_tag = mml_music_tag_df[mml_music_tag_df['tag'].str.contains(tag)]
            music_list.append(music_with_tag)

        # 모든 결과를 하나의 DataFrame으로 결합
        similar_mml_music_tag_df = pd.concat(music_list).drop_duplicates().reset_index(drop=True)

        genre_reference_df = mml_music_info_df[['title', 'artist', 'genre']].copy()

        # We create a similar lowercase version of Title and Artist in music_tag for a case-insensitive merge
        music_tag_lowercase = similar_mml_music_tag_df[['title', 'artist', 'tag']].copy()

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
        # user_id = 'QrDM6lLc'
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

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        tag_song2vec_final = pd.merge(
            mml_music_info_df, recommendations_with_similarity,
            on=['title', 'artist'],
            how='inner'
        )

        tag_song2vec_final = tag_song2vec_final[['title', 'artist', 'album_image_url']]

        tag_song2vec_results=[]

        for index,row in tag_song2vec_final.iterrows():
            result = {
                'title': row['title'],
                'artist': row['artist'],
                'image': row['album_image_url']
            }
            tag_song2vec_results.append(result)

        return Response(tag_song2vec_results, status=status.HTTP_200_OK)