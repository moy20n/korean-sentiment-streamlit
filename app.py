import streamlit as st 
import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 데이터 로드 및 전처리 ---
df = pd.read_csv("korean_sentiment2.csv")
df = df.dropna(subset=['text', 'label'])  
df['text'] = df['text'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()

# --- 모델 학습 ---
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df['text'], df['label'])

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="감정 분석 AI", page_icon="💙", layout="centered")

# --- CSS 스타일 ---
st.markdown("""
<style>
/* 전체 배경 */
.main {
  background-color: rgba(204, 229, 255, 0.35);
  font-family: 'Pretendard', sans-serif;
  padding: 30px 50px 50px 50px;
  min-height: 100vh;
}

/* 제목 */
h1 {
  color: #0059CC;
  font-weight: 800;
  text-align: center;
  margin-bottom: 12px;
  font-size: 48px;
  letter-spacing: 1.2px;
}

/* 부제목 */
p {
  color: #0073E6;
  text-align: center;
  margin-top: 0;
  margin-bottom: 48px;
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 0.7px;
}

/* 텍스트 박스 */
.stTextArea > div > textarea {
  background-color: rgba(229, 244, 255, 0.7);
  font-size: 18px;
  border-radius: 16px;
  padding: 20px;
  color: #004A99;
  font-weight: 600;
  border: 1.8px solid rgba(0, 102, 204, 0.3);
  min-height: 180px;
  max-width: 750px;
  margin: 0 auto;
  display: block;
  resize: vertical;
  box-shadow: 0 0 12px rgba(0, 102, 204, 0.15);
}

/* 버튼 - 가운데 정렬 */
div.stButton {
  display: flex;
  justify-content: center;
  margin-bottom: 50px;
}
div.stButton > button:first-child {
  background-color: #FFFFFF;
  color: #FFFFFF;
  border: none;
  border-radius: 14px;
  padding: 0.85em 2em;
  font-size: 20px;
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.35s ease;
  box-shadow: none;
  min-width: 260px;
}
div.stButton > button:first-child:hover {
  background-color: #66B2FF;
  color: #FFFFFF;
}

/* 결과 박스 */
.result-box {
  background-color: rgba(204, 229, 255, 0.8);
  max-width: 750px;
  margin: 0 auto 50px auto;
  padding: 35px 25px;
  border-radius: 18px;
  text-align: center;
  box-shadow: 0 0 24px rgba(51, 153, 255, 0.15);
  color: #004A99;
  font-weight: 700;
  font-size: 24px;
}

/* 사이드바 배경 및 텍스트 */
[data-testid="stSidebar"] {
  background-color: #F5F8FF;
  color: #003366;
  padding: 20px 20px 30px 20px;
  font-family: 'Pretendard', sans-serif;
  border-right: 1px solid #CCE5FF;

  /* 너비 조정 추가 */
  width: 320px !important;
  min-width: 320px !important;
}

/* 사이드바 제목 스타일 */
[data-testid="stSidebar"] h2 {
  color: #3399FF;
  font-weight: 700;
  font-size: 24px;
  margin-bottom: 14px;
  letter-spacing: 1.0px;
  text-align: center;
}

/* 사이드바 텍스트 */
[data-testid="stSidebar"] p {
  color: #004A99;
  font-weight: 500;
  font-size: 15px;
  line-height: 1.5;
  margin-top: 0;
  text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- 사이드바 ---
st.sidebar.markdown("<h2>✨감정 분석기✨</h2>", unsafe_allow_html=True)
st.sidebar.markdown("한글 문장을 입력하면 감정을 분석해드려요! ☘\n\nMade by 호연")

# --- 메인 UI ---
st.markdown('<h1>🌊 한글 감정 분석 AI 🌊</h1>', unsafe_allow_html=True)
st.markdown('<p>문장을 입력하면 감정을 분석해드려요! 🍀</p>', unsafe_allow_html=True)

text = st.text_area("👇 감정을 알고 싶은 문장을 입력해 주세요:", height=180, placeholder="예) 오늘은 너무 행복해요! 🌠")

if st.button("🍀 감정 분석하기 🍀"):
    if text.strip() == "":
        st.warning("⚠️ 문장을 입력해 주세요!")
    else:
        result = model.predict([text])[0]

        style_map = {
            "긍정": ("😊", "긍정"),
            "부정": ("😢", "부정"),
            "중립": ("😐", "중립")
        }
        emoji, label = style_map.get(result, ("🤔", "알 수 없음"))

        st.markdown(
            f"""
            <div class="result-box">
                예측 감정: {label} {emoji}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("💡 *Streamlit과 Naive Bayes로 구현된 간단한 감정 분석기입니다.*")
