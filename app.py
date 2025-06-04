import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# 데이터 로딩 캐시 (데이터셋은 온라인에서 바로 불러옴)
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

# 모델 학습 및 캐시
@st.cache_resource(show_spinner=False)
def train_model(data):
    X = data.drop(columns='quality')
    y = data['quality']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'wine_quality_model.pkl')
    return model

st.title("와인 품질 예측기 (Red Wine) 🍷")
st.write("와인의 화학적 특성을 입력하면 품질 점수를 예측합니다.")

# 데이터 불러오기
data = load_data()

# 슬라이더 생성 함수: 각 특성별 최소, 최대, 평균값을 활용
def slider_with_range(feature):
    min_val = float(data[feature].min())
    max_val = float(data[feature].max())
    mean_val = float(data[feature].mean())
    return st.slider(f"{feature}", min_val, max_val, mean_val)

# 사용자 입력값 받기
fixed_acidity = slider_with_range('fixed acidity')
volatile_acidity = slider_with_range('volatile acidity')
citric_acid = slider_with_range('citric acid')
residual_sugar = slider_with_range('residual sugar')
chlorides = slider_with_range('chlorides')
free_sulfur_dioxide = slider_with_range('free sulfur dioxide')
total_sulfur_dioxide = slider_with_range('total sulfur dioxide')
density = slider_with_range('density')
pH = slider_with_range('pH')
sulphates = slider_with_range('sulphates')
alcohol = slider_with_range('alcohol')

input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])

# 모델 불러오기 또는 학습
try:
    model = joblib.load('wine_quality_model.pkl')
except FileNotFoundError:
    model = train_model(data)

# 예측
prediction = model.predict(input_features)[0]

st.markdown(f"### 예측 와인 품질 점수: **{prediction}점**")
