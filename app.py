import streamlit as st 
import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 📦 데이터 로드
df = pd.read_csv("korean_sentiment2.csv")

# 🧹 NaN 및 공백 처리
df = df.dropna(subset=['text', 'label'])  
df['text'] = df['text'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()

# 🧠 모델 학습
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df['text'], df['label'])

# 🎨 Streamlit UI 꾸미기
st.set_page_config(page_title="감정 분석 AI", page_icon="🧠", layout="centered")

# 🎉 사이드바
st.sidebar.title("✨ 감정 분석기 ✨")
st.sidebar.markdown("한글 문장을 입력하면 감정을 분석해드려요! 😊\n\nMade with ❤️ by 호연")

# 💬 메인 영역
st.title("💡 한글 감정 분석 AI 🔍")
st.markdown("문장을 입력하면 감정을 분석해드릴게요! 😎")

text = st.text_area("👇 여기에 문장을 입력해 주세요", height=150, placeholder="예: 오늘은 기분이 너무 좋아! ☀️")

# 🎯 버튼 클릭 시 분석 실행
if st.button("✨ 감정 분석하기 ✨"):
    if text.strip() == "":
        st.warning("⚠️ 문장을 입력해 주세요!")
    else:
        result = model.predict([text])[0]

        # 🎭 감정별 이모지 및 색상 설정
        emojis = {
            "긍정": "😊💖🎈",
            "부정": "😢💔🌧️",
            "중립": "😐📘🍃"
        }

        colors = {
            "긍정": "#d0f0c0",   # 연두
            "부정": "#fcdede",   # 연핑
            "중립": "#e0e0e0"    # 회색
        }

        # 🎨 배경색 효과
        st.markdown(
            f"""
            <div style='background-color:{colors.get(result, "#f0f0f0")}; 
                        padding:20px; border-radius:12px; text-align:center'>
                <h2>예측 감정: {result} {emojis.get(result, '🤔')}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.balloons()  # 🎈🎈

# 🔗 하단 메시지
st.markdown("---")
st.markdown("💡 *Streamlit과 Naive Bayes로 구현된 간단한 감정 분석기입니다.*")
