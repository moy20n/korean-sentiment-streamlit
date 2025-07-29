import streamlit as st 
import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 기본 설정 ---
st.set_page_config(page_title="감정 분석기 💬", page_icon="💙", layout="centered")

# --- CSS로 스타일 꾸미기 ---
st.markdown("""
    <style>
        .title {
            font-size:40px;
            font-weight:bold;
            color:#2E86C1;
            text-align:center;
        }
        .subtitle {
            font-size:18px;
            color:#555;
            text-align:center;
            margin-bottom:20px;
        }
        .stTextArea > div > textarea {
            background-color: #F0F8FF;
            font-size: 16px;
        }
        div.stButton > button:first-child {
            background-color: #87CEFA;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #00BFFF;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- 데이터 로드 ---
df = pd.read_csv("korean_sentiment2.csv")

# NaN 및 공백 처리
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()

# --- 모델 학습 ---
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df['text'], df['label'])

# --- Streamlit UI ---
st.markdown('<div class="title">감정 분석 AI 😄😢😠</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">입력한 문장의 감정을 예측해 드립니다 💌</div>', unsafe_allow_html=True)

text = st.text_area("💬 감정을 알고 싶은 문장을 입력해 주세요:")

if st.button("🔍 분석하기"):
    if text.strip():
        result = model.predict([text])[0]
        emoji = "😊" if result == "positive" else "😢" if result == "negative" else "😐"
        st.markdown(f"### 예측 감정: <span style='color:#2E86C1; font-size:24px'>**{result}**</span> {emoji}", unsafe_allow_html=True)
    else:
        st.warning("문장을 입력해주세요! 🙏")
