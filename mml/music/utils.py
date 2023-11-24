import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_top_words_weights(lyrics_list, top_n=20):
            # 모든 가사를 하나의 리스트로 결합합니다.
            all_words = [word for lyrics in lyrics_list for word in lyrics]
            # 가장 흔한 단어와 그 빈도수를 계산합니다.
            top_words = pd.Series(all_words).value_counts().head(top_n)
            # 가중치를 계산합니다: 여기서는 단순화를 위해 빈도수를 그대로 사용하지만,
            # 다른 가중치 할당 방식을 사용할 수도 있습니다.
            weights = top_words / top_words.max()  # 최대 빈도수로 정규화하여 가중치를 계산합니다.
            return weights.to_dict()

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

# 태그 데이터를 전처리하는 함수를 정의합니다.
def preprocess_tags(tag_string):
    # '#' 기호를 기준으로 태그를 분리합니다.
    tags = tag_string.strip().split('#')
    # 빈 문자열을 제거합니다.
    tags = [tag for tag in tags if tag]  # 공백 태그 제거
    return tags

# 태그를 벡터로 변환하는 함수를 정의합니다.
def vectorize_tags(tags, w2v_model):
    tag_vectors = []
    for tag in tags:
        # 태그 내의 각 단어에 대해 벡터를 얻고 평균을 계산합니다.
        tag_word_vectors = [w2v_model.wv[word] for word in tag.split() if word in w2v_model.wv]
        if tag_word_vectors:  # 태그가 모델 단어장에 있는 경우에만 평균 벡터를 계산합니다.
            tag_vectors.append(np.mean(tag_word_vectors, axis=0))
    return np.mean(tag_vectors, axis=0) if tag_vectors else np.zeros(w2v_model.vector_size)

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