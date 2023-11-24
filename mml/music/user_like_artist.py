# from django.shortcuts import render
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# # 기타 필요한 import 문
# from sqlalchemy import create_engine
# from data_loader import DataLoader
# from sklearn.metrics.pairwise import cosine_similarity
# import random

# engine = create_engine('mysql+pymysql://admin:pizza715@mml.cu4cw1rqzfei.ap-northeast-2.rds.amazonaws.com/mml?charset=utf8')

# # DataLoader 인스턴스 생성
# data_loader = DataLoader(engine)

# mml_user_his_df, mml_music_info_df, mml_music_tag_df, mml_artist_gen_df, mml_user_like_artist_df = data_loader.load_data()

# user_id = '08XxwFym'

# class user_like_artist_view(APIView):
#     def get(self, request):
#         print('2번')

#         # 데이터 전처리
#         # 사용자가 좋아하는 아티스트 데이터와 아티스트 장르 데이터를 병합하여 좋아하는 아티스트의 장르를 구합니다.
#         merged_data = pd.merge(mml_user_like_artist_df, mml_artist_gen_df, on='artist', how='left')

#         # 사용자별로 데이터를 그룹화하고 좋아하는 모든 장르, 성별, 연령 그룹을 연결합니다.
#         user_genre_df = merged_data.groupby('user_id').agg({
#             'genre': lambda x: ' '.join(x.dropna()),
#             'gen': 'first',    # 가정: 모든 행에 대해 동일한 값이 존재한다고 가정
#             'age_group': 'first'  # 가정: 모든 행에 대해 동일한 값이 존재한다고 가정
#         }).reset_index()


#         # ITF-IDF 벡터 구현
#         tfidf = TfidfVectorizer(stop_words='english')
#         tfidf_matrix = tfidf.fit_transform(user_genre_df['genre'])

#         # 코사인 유사도 계산
#         cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#         # 사용자 ID를 기반으로 노래를 추천하는 기능 (성별 및 연령대 필터링 적용)
#         def recommend_songs(user_id, num_recommendations=3):
#             # 사용자 id와 동일한 성별 및 연령대를 가진 사용자들만 필터링
#             target_user_data = user_genre_df[user_genre_df['user_id'] == user_id]
#             if target_user_data.empty:
#                 return "사용자 ID를 찾을 수 없습니다."

#             target_gen = target_user_data['gen'].iloc[0]
#             target_age_group = target_user_data['age_group'].iloc[0]

#             filtered_users = user_genre_df[(user_genre_df['gen'] == target_gen) &
#                                         (user_genre_df['age_group'] == target_age_group)]

#             # 필터링된 사용자들의 인덱스 추출
#             filtered_user_indices = filtered_users.index.tolist()

#             # 해당 사용자와 필터링된 사용자들의 유사성 점수를 가져옵니다.
#             idx = user_genre_df.index[user_genre_df['user_id'] == user_id].tolist()[0]
#             sim_scores = list(enumerate(cosine_sim[idx]))

#             # 유사성 점수 중에서 필터링된 사용자들만을 대상으로 정렬
#             sim_scores = [sim_score for sim_score in sim_scores if sim_score[0] in filtered_user_indices]
#             sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#             # 가장 유사한 사용자들의 점수
#             sim_scores = sim_scores[1:num_recommendations+1]

#             # 사용자 인덱스
#             user_indices = [i[0] for i in sim_scores]

#             # 가장 유사한 사용자들 반환
#             return user_genre_df['user_id'].iloc[user_indices]
        
#         recommended_users = recommend_songs(user_id)

#         # 추천 함수를 수정하여 이미 선호하는 아티스트를 제외하고 추천
#         def recommend_new_artists(user_id, num_recommendations=10):

#             # 원래 사용자의 아티스트 선호도 가져오기
#             user_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'] == user_id]['artist'].tolist()

#             # 사용자 id에 기반한 추천 받기
#             recommended_user_ids = recommend_songs(user_id, num_recommendations).tolist()

#             # 추천된 사용자들이 선호하는 아티스트 찾기
#             recommended_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'].isin(recommended_user_ids)]['artist']

#             # 원래 사용자가 선호하는 아티스트 제외
#             new_recommended_artists = recommended_artists[~recommended_artists.isin(user_artists)].unique()

#             return new_recommended_artists

#         # 이제 추천 함수를 다시 실행하여 테스트 사용자와 유사한 사용자를 찾을 수 있습니다.
#         recommended_user_ids = recommend_songs(user_id).tolist()

#         # 유사한 사용자들이 선호하는 아티스트 찾기
#         preferred_artists = mml_user_like_artist_df[mml_user_like_artist_df['user_id'].isin(recommended_user_ids)]['artist'].unique()

#         def get_all_songs_for_artists(artist_list):
#             songs_dict = {}
#             for artist in artist_list:
#                 # 해당 아티스트의 모든 노래만 필터링
#                 artist_songs = mml_music_info_df[mml_music_info_df['artist'] == artist]
#                 # 노래가 존재하는 경우에만 딕셔너리에 추가
#                 if not artist_songs.empty:
#                     songs_dict[artist] = artist_songs['title'].tolist()
#             return songs_dict

#         # 모든 추천된 아티스트별로 모든 노래를 가져옵니다.
#         artist_songs_dict = get_all_songs_for_artists(preferred_artists)

#         # 랜덤으로 20곡 선택하기 전에 각 곡에 해당하는 아티스트 정보도 포함시키기
#         all_songs_with_artists = []
#         for artist, songs in artist_songs_dict.items():
#             for song in songs:
#                 all_songs_with_artists.append((artist, song))

#         # 랜덤으로 20곡 선택 (아티스트 정보 포함)
#         selected_songs_with_artists = random.sample(all_songs_with_artists, 20)

#         # 선택된 노래와 아티스트를 데이터프레임으로 변환
#         df_selected_songs_with_artists = pd.DataFrame(selected_songs_with_artists, columns=['artist', 'title'])

#         # Merge the dataframes on 'Title' and 'Artist' to find matching songs
#         user_like_artist_final = pd.merge(
#             mml_music_info_df, df_selected_songs_with_artists,
#             on=['title', 'artist'],
#             how='inner'
#         )

#         user_like_artist_final = user_like_artist_final[['title', 'artist', 'album_image_url']]

#         user_like_artist_results=[]

#         for index,row in user_like_artist_final.iterrows():
#             result = {
#                 'title': row['title'],
#                 'artist': row['artist'],
#                 'image': row['album_image_url']
#             }
#             user_like_artist_results.append(result)

#         return Response(user_like_artist_results, status=status.HTTP_200_OK)
