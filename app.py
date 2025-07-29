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
/* 전체 배경 - 투명하고 맑은 바다 느낌 파스텔 블루 */
.main {
  background-color: rgba(204, 229, 255, 0.35); /* 반투명 연한 하늘색 */
  font-family: 'Pretendard', sans-serif;
  padding: 30px 50px 50px 50px;
  min-height: 100vh;
}

/* 제목 */
h1 {
  color: #0059CC;  /* 맑고 선명한 청색 */
  font-weight: 800;
  text-align: center;
  margin-bottom: 12px;
  font-size: 50px;
  letter-spacing: 1.2px;
}

/* 부제목 */
p {
  color: #0073E6;
  text-align: center;
  margin-top: 0;
  margin-bottom: 48px;
  font-size: 22px;
  font-weight: 600;
  letter-spacing: 0.7px;
}

/* 텍스트 박스 */
.stTextArea > div > textarea {
  background-color: rgba(229, 244, 255, 0.7);  /* 투명하고 부드러운 바다색 */
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

/* 버튼 */
div.stButton > button:first-child {
  background-color: #3399FF;  /* 청명한 파란색 */
  color: #FFFFFF;  /* 하얀 글씨 */
  border: none;
  border-radius: 14px;
  padding: 0.85em 2em;
  font-size: 20px;
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.35s ease, box-shadow 0.35s ease;
  box-shadow: 0 0 14px #66B2FF;
  display: block;
  margin: 30px auto 50px auto;
  min-width: 260px;
}
div.stButton > button:first-child:hover {
  background-color: #66B2FF;  /* 밝은 파랑 */
  box-shadow: 0 0 22px #99CCFF;
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
  box-shadow: 0 0 24px rgba(51, 153, 255, 0.3);
}

/* 반짝임 글자 */
.glow-text {
  font-size: 46px;
  font-weight: 900;
  color: #0073E6;
  text-align: center;
  text-shadow:
    0 0 5px rgba(0, 115, 230, 0.65),
    0 0 8px rgba(51, 153, 255, 0.45),
    0 0 14px rgba(102, 178, 255, 0.3);
  animation: borderGlow 4s ease-in-out infinite alternate;
  margin-bottom: 0;
}

/* 부드러운 반짝임 애니메이션 */
@keyframes borderGlow {
  0%, 100% {
    text-shadow:
      0 0 4px rgba(0, 115, 230, 0.5),
      0 0 7px rgba(51, 153, 255, 0.35),
      0 0 10px rgba(102, 178, 255, 0.2);
  }
  50% {
    text-shadow:
      0 0 10px rgba(0, 115, 230, 1),
      0 0 15px rgba(51, 153, 255, 0.85),
      0 0 25px rgba(102, 178, 255, 0.6);
  }
}

/* 사이드바 배경 및 텍스트 */
[data-testid="stSidebar"] {
  background-color: #F5F8FF;  /* 아주 연한 하늘색 */
  color: #003366;
  padding: 20px 20px 30px 20px;
  font-family: 'Pretendard', sans-serif;
  border-right: 1px solid #CCE5FF;
}

/* 사이드바 제목 */
[data-testid="stSidebar"] h2 {
  color: #004080;
  font-weight: 700;
  font-size: 22px;
  margin-bottom: 14px;
  letter-spacing: 0.8px;
}

/* 사이드바 텍스트 */
[data-testid="stSidebar"] p {
  color: #004A99;
  font-weight: 500;
  font-size: 15px;
  line-height: 1.5;
  margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

# --- 메인 UI ---
st.markdown('<h1>💡 한글 감정 분석 AI 🔍</h1>', unsafe_allow_html=True)
st.markdown('<p>문장을 입력하면 감정을 분석해드려요! 😎</p>', unsafe_allow_html=True)

text = st.text_area("👇 감정을 알고 싶은 문장을 입력해 주세요:", height=180, placeholder="예) 오늘은 너무 행복해요! 🌞")

if st.button("✨ 감정 분석하기 ✨"):
    if text.strip() == "":
        st.warning("⚠️ 문장을 입력해 주세요!")
    else:
        result = model.predict([text])[0]

        style_map = {
            "긍정": ("glow-text", "😊💖🎈", "rgba(204, 229, 255, 0.9)"),
            "부정": ("glow-text", "😢💔🌧️", "rgba(179, 198, 225, 0.85)"),
            "중립": ("glow-text", "😐📘🍃", "rgba(194, 210, 236, 0.85)")
        }
        css_class, emoji, bg_color = style_map.get(result, ("glow-text", "🤔", "rgba(230, 240, 255, 0.85)"))

        st.markdown(
            f"""
            <div class="result-box" style="background-color:{bg_color};">
                <div class="{css_class}">
                    예측 감정: {result} {emoji}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("💡 *Streamlit과 Naive Bayes로 구현된 간단한 감정 분석기입니다.*")
