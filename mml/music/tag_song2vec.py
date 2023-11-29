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

import Levenshtein as lev

from scipy.stats import pearsonr

from sqlalchemy import create_engine

from django.apps import apps

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from song2vec_data_loader import song2vec_DataLoader

from django.contrib.sessions.models import Session
from user.models import MMLUserInfo
from .models import MMLMusicTagHis

engine = create_engine('mysql+pymysql://admin:pizza715@mml.cu4cw1rqzfei.ap-northeast-2.rds.amazonaws.com/mml?charset=utf8')

# DataLoader 인스턴스 생성
song2vec_data_loader = song2vec_DataLoader(engine)

mml_user_his_df, mml_music_info_df, mml_music_tag_df, music_data, music_tag_data = song2vec_data_loader.song2vec_load_data()

class tag_song2vec_view(APIView):
    def get(self, request):
        print('==========4번==========')
        
        session_key = request.COOKIES.get("sessionid")
    
        if session_key:
            try:
                # Retrieve the session object from the database
                session = Session.objects.get(session_key=session_key)
                # Get the decoded session data
                session_data = session.get_decoded()
                # Print the session data
                print("Session Data:", session_data)
                session_id = session_data.get("_auth_user_id")
                # Access specific values from the session data
                user = MMLUserInfo.objects.get(pk=session_id)
                user_id = str(user)
                print("User ID from session:", user_id)
                
                # Other logic in your view
            except Session.DoesNotExist:
                print("Session with key does not exist")
        else:
            print("Session key does not exist")

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

        # Okt 객체 생성
        okt = Okt()

        # 불용어 목록 정의
        stopwords = set(['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다'])

        def find_similar_tags(word, tag_list, threshold=0.5):
            # 주어진 단어와 태그 목록 사이의 유사한 태그를 찾는 함수
            similar_tags = []
            for tag in tag_list:
                similarity = lev.ratio(word, tag)
                if similarity >= threshold:
                    similar_tags.append((tag, similarity))
            return similar_tags

        # 문장에서 형태소 추출 및 불용어 제거
        # 사용자로부터의 입력 문장
        input_sentence = request.GET.get('input_sentence', None)
        print("input_sentence의 값 : ", input_sentence)
        print("input_sentence의 타입:", type(input_sentence))
        if not input_sentence:
            return Response({'error': 'input_sentence가 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)
        
        morphs = [morph for morph in okt.morphs(input_sentence) if morph not in stopwords]

        # 태그 목록
        tags = [
            "감성적인", "팝송", "밤", "새벽", "카페", "휴식", "명상", "드라이브", "신나는", "가을", "인디", "그루브한", "발라드한", "기분전환", "경쾌한", "여름", "겨울", "힐링", "잔잔한", "연주곡", "재즈", "산책", "여행", "R&B", "사랑", "기쁨", "스트레스", "짜증", "공연", "페스티벌", "봄", "그리움", "일상", "저녁", "이별", "슬픔", "화창한", "힙합", "락", "비", "흐림", "KPOP", "아이돌", "등교", "일렉트로닉", "청량한", "달달한", "운동", "헬스", "오후", "올디스", "페스티벌", "공부", "감각적인", "사무실", "아침", "쌀쌀한", "지치고", "힘들때", "우울할때", "출근", "방안에서", "2010년대", "새벽감성", "편집숍", "매장", "클럽", "파티", "독서", "어쿠스틱한", "선선한", "목소리", "음색", "OST", "술집", "펍", "폭염", "더위", "애절한", "뉴에이지", "일렉트로닉팝", "크리스마스", "클래식", "공연", "라이브", "잠들기전", "몽환적인", "쓸쓸한", "설렘", "심쿵", "미세먼지", "황사", "혼술", "혼밥", "맑음", "강한", "환절기", "소울풀한", "싱숭생숭", "시원한", "외로울때", "울고", "눈", "상쾌한", "EDM", "호텔", "바", "고백", "멘붕", "불안", "썸", "듀엣", "피처링", "섹시한", "1990년대", "트로트", "노래방", "나들이", "소풍", "걸그룹", "2000년대", "2020년대", "리메이크", "하교", "퇴근", "JPOP", "1980년대", "월드뮤직", "1970년대", "보사노바", "뉴에이지", "BGM", "동요", "키즈", "청춘", "결혼", "CCM", "태교", "릴스", "1960년대", "클래식", "답답할때", "10대", "20대", "패션쇼", "뮤지컬", "국악", "합창", "광고"
        ]

        # 각 형태소에 대해 유사한 태그 찾기 및 유사도 순으로 정렬
        similar_tags_for_morphs = {}
        for morph in morphs:
            similar_tags = find_similar_tags(morph, tags)
            # 유사도 점수에 따라 태그 정렬 (내림차순)
            similar_tags.sort(key=lambda x: x[1], reverse=True)
            if similar_tags:
                similar_tags_for_morphs[morph] = similar_tags[0][0]

        # 모든 태그를 포함하는 음악만 필터링
        similar_mml_music_tag_df = mml_music_tag_df
        for tag in similar_tags_for_morphs.values():
            similar_mml_music_tag_df = similar_mml_music_tag_df[similar_mml_music_tag_df['tag'].str.contains(tag)]

        genre_reference_df = mml_music_info_df[['title', 'artist', 'genre']].copy()

        # 대소문자를 구분하지 않는 병합을 위해 'music_tag'의 'Title' 및 'Artist' 열에 대한 소문자 버전을 만듭니다.
        music_tag_lowercase = similar_mml_music_tag_df[['title', 'artist', 'tag']].copy()

        # 'genre_reference_df' 데이터프레임을 'music_tag'와 일치하는 'Title' 및 'Artist'를 대소문자 구분 없이 병합합니다.
        music_tag_data = pd.merge(music_tag_lowercase, genre_reference_df,
                                        left_on=['title', 'artist'], right_on=['title', 'artist'],
                                        how='left')

        def get_top_words_weights(lyrics_list, top_n=20):
            # 모든 가사를 하나의 리스트로 결합합니다.
            all_words = [word for lyrics in lyrics_list for word in lyrics]
            # 가장 흔한 단어와 그 빈도수를 계산합니다.
            top_words = pd.Series(all_words).value_counts().head(top_n)
            # 중복을 제거한 단어 리스트를 생성합니다.
            unique_words = set(top_words.index)
            # 가중치를 계산합니다: 여기서는 단순화를 위해 빈도수를 그대로 사용하지만,
            # 다른 가중치 할당 방식을 사용할 수도 있습니다.
            weights = {word: freq / top_words.max() for word, freq in top_words.items() if word in unique_words}
            return weights
        
        # 사용자의 가사 프로필을 만들 때, 가장 흔한 단어에 가중치를 주어 벡터를 계산하는 함수를 수정합니다.
        def create_weighted_lyrics_profile(lyrics_list, w2v_model, top_words_weights):
            lyrics_vectors = []
            for lyrics in lyrics_list:
                # lyrics 벡터의 평균을 계산하기 전에 각 단어에 대한 가중치를 고려합니다.
                weighted_vectors = []
                for word in lyrics:
                    if word in w2v_model.wv:
                        weight = top_words_weights.get(word, 1)  # 여기서 get 메소드를 사용
                        weighted_vectors.append(w2v_model.wv[word] * weight)
                if weighted_vectors:  # 가중치가 적용된 벡터의 평균을 계산합니다.
                    lyrics_vectors.append(np.mean(weighted_vectors, axis=0))
            return np.mean(lyrics_vectors, axis=0) if lyrics_vectors else np.zeros(w2v_model.vector_size)

        # 사용자별 프로필 벡터를 생성합니다.
        user_lyrics = music_data[music_data['user'] == user_id]['processed_lyrics']
        # 가사 리스트를 사용하여 가중치 사전을 생성
        weights_dict = get_top_words_weights(user_lyrics)
        # 사용자의 가사 프로필을 생성
        user_profile_vector = create_weighted_lyrics_profile(user_lyrics, w2v_model, weights_dict)
        # user_profile_vector = create_lyrics_profile(user_lyrics, w2v_model)


        # 태그 데이터를 전처리하는 함수를 정의합니다.
        def preprocess_tags(tag_string):
            # '#' 기호를 기준으로 태그를 분리합니다.
            tags = tag_string.strip().split('#')
            # 빈 문자열을 제거합니다.
            tags = [tag for tag in tags if tag]  # 공백 태그 제거
            return tags

        # 태그 데이터에 전처리 함수를 적용합니다.
        music_tag_data['processed_tags'] = music_tag_data['tag'].apply(preprocess_tags)

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
        music_tag_data['tag_vector'] = music_tag_data['processed_tags'].apply(lambda tags: vectorize_tags(tags, w2v_model))

        # 사용자 프로필 벡터와 모든 태그 벡터 사이의 코사인 유사도를 계산하고 상위 N개의 추천과 함께 유사도를 반환하는 함수
        def recommend_songs_with_similarity(user_profile_vector, tag_vectors, songs_data, top_n=20):
            # 사용자 프로필 벡터를 코사인 유사도 계산을 위해 reshape
            user_vector_reshaped = user_profile_vector.reshape(1, -1)

            # 모든 태그 벡터와의 유사도 계산
            similarity_scores = cosine_similarity(user_vector_reshaped, tag_vectors)[0]

            # 유사도 점수를 기반으로 상위 N개의 인덱스를 가져옵니다
            top_indices = similarity_scores.argsort()[-top_n:][::-1]
            print(top_indices)
            # 상위 N개의 노래 추천 정보와 유사도 점수를 함께 반환
            recommendations_with_scores = songs_data.iloc[top_indices]
            recommendations_with_scores['similarity'] = similarity_scores[top_indices]
            return recommendations_with_scores[['title', 'artist', 'tag', 'similarity']]

        # 모든 태그 벡터를 하나의 배열로 추출합니다.
        tag_vectors_matrix = np.array(list(music_tag_data['tag_vector']))

        # 사용자 ID에 대한 노래 추천을 받고 유사도 점수를 포함하여 출력합니다.
        # user_profile_vector_for_similarity = user_profiles[user_id_to_recommend]  # 해당 사용자의 프로필 벡터를 가져옵니다.
        recommendations_with_similarity = recommend_songs_with_similarity(user_profile_vector, tag_vectors_matrix, music_tag_data)
        print(recommendations_with_similarity)

        # Merge the dataframes on 'Title' and 'Artist' to find matching songs
        tag_song2vec_final = pd.merge(
            mml_music_info_df, recommendations_with_similarity,
            on=['title', 'artist'],
            how='inner'
        )
        

        
        tag_song2vec_final = tag_song2vec_final[['title', 'artist', 'album_image_url']]
        tag_song2vec_final['user_id'] = user_id
        tag_song2vec_final['input_sentence'] = input_sentence
        
        
        tag_song2vec_results = []
        for index, row in tag_song2vec_final.iterrows():
            MMLMusicTagHis.objects.create(
                title=row['title'],
                artist=row['artist'],
                image=row['album_image_url'],
                user_id=user_id,  # 이 값은 어딘가에서 가져와야 함
                input_sentence=input_sentence  # 이 값은 어딘가에서 가져와야 함
            )

        # 결과를 JSON 형식으로 반환
        return Response(tag_song2vec_results, status=status.HTTP_200_OK)