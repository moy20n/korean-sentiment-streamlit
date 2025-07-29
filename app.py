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

# --- 사이드바 ---
st.sidebar.title("✨ 감정 분석기 Ver. ChatGPT ✨")
st.sidebar.markdown("한글 문장을 입력하면 감정을 분석해드려요! 😊\n\nMade with ❤️ by 호연")

# --- CSS 스타일 ---
st.markdown("""
<style>
/* 배경색 연한 파스텔 블루 */
.main {
  background-color: #E6F0FF;  /* 밝고 파란 파스텔 하늘색 */
  font-family: 'Pretendard', sans-serif;
  padding: 20px 40px 40px 40px;
}

/* 제목 스타일 */
h1 {
  color: #1A4DFF;  /* 파란 느낌 더 강한 색상 */
  font-weight: 800;
  text-align: center;
  margin-bottom: 8px;
  font-size: 48px;
}

/* 부제목 스타일 */
p {
  color: #3A6EFF;
  text-align: center;
  margin-top: 0;
  margin-bottom: 40px;
  font-size: 20px;
  font-weight: 600;
}

/* 텍스트 박스 스타일 */
.stTextArea > div > textarea {
  background-color: #CDE1FF;  /* 부드러운 하늘색 */
  font-size: 18px;
  border-radius: 14px;
  padding: 15px;
  color: #1D3CCC;
  font-weight: 600;
  border: 1.5px solid #A3BFF7;
  min-height: 180px;
  max-width: 700px;
  margin: 0 auto;
  display: block;
  resize: vertical;
}

/* 버튼 스타일 */
div.stButton > button:first-child {
  background-color: #82A9FF;  /* 파스텔 파란색 */
  color: #FFFFFF;  /* 하얀 글씨 */
  border: none;
  border-radius: 12px;
  padding: 0.8em 1.8em;
  font-size: 20px;
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.3s ease;
  box-shadow: 0 0 12px #8AB4FF;
  display: block;
  margin: 20px auto 40px auto;
  min-width: 240px;
}
div.stButton > button:first-child:hover {
  background-color: #A3C2FF;  /* 더 밝은 파란색 */
  box-shadow: 0 0 18px #94B8FF;
  color: #FFFFFF;
}

/* 결과 박스 스타일 */
.result-box {
  background-color: #D6E4FF;
  max-width: 700px;
  margin: 0 auto 40px auto;
  padding: 30px 20px;
  border-radius: 16px;
  text-align: center;
  box-shadow: 0 0 18px rgba(130, 180, 255, 0.5);
}

/* 반짝임 글자 스타일 */
.glow-text {
  font-size: 44px;
  font-weight: 800;
  color: #2355FF;
  text-align: center;
  text-shadow:
    0 0 5px rgba(56, 102, 255, 0.6),
    0 0 8px rgba(90, 140, 255, 0.4),
    0 0 12px rgba(130, 180, 255, 0.3);
  animation: borderGlow 3.5s ease-in-out infinite alternate;
  margin-bottom: 0;
}

/* 부드러운 반짝임 애니메이션 */
@keyframes borderGlow {
  0%, 100% {
    text-shadow:
      0 0 3px rgba(56, 102, 255, 0.5),
      0 0 5px rgba(90, 140, 255, 0.3),
      0 0 7px rgba(130, 180, 255, 0.2);
  }
  50% {
    text-shadow:
      0 0 7px rgba(56, 102, 255, 1),
      0 0 10px rgba(90, 140, 255, 0.8),
      0 0 15px rgba(130, 180, 255, 0.6);
  }
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

        # 감정별 CSS 클래스, 이모지, 배경색 설정
        style_map = {
            "긍정": ("glow-text", "😊💖🎈", "#E6EEFF"),
            "부정": ("glow-text", "😢💔🌧️", "#D4DAF8"),
            "중립": ("glow-text", "😐📘🍃", "#E0E6F9")
        }
        css_class, emoji, bg_color = style_map.get(result, ("glow-text", "🤔", "#F0F4FF"))

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
