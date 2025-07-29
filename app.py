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

# --- CSS 스타일 + 파스텔 하늘색 테마 + 부드러운 테두리 반짝임 + 버튼 글자 하얀색 ---
st.markdown("""
<style>
/* 기본 배경 */
.main {
  background-color: #F7FAFF;  /* 아주 연한 파스텔 하늘색 */
  font-family: 'Pretendard', sans-serif;
}

/* 텍스트 영역 배경 */
.stTextArea > div > textarea {
  background-color: #DDE7FF;  /* 부드러운 파스텔 하늘색 */
  font-size: 16px;
  border-radius: 10px;
  padding: 10px;
  color: #5A75C9;
  font-weight: 600;
  border: 1.5px solid #B0BDEB;
}

/* 버튼 꾸미기 */
div.stButton > button:first-child {
  background-color: #A9C1FF;  /* 부드러운 파스텔 하늘색 */
  color: #FFFFFF;  /* 하얀색으로 변경 */
  border: none;
  border-radius: 8px;
  padding: 0.6em 1.2em;
  font-size: 18px;
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.3s ease;
  box-shadow: 0 0 8px #B0BDEB;
}
div.stButton > button:first-child:hover {
  background-color: #C3D1FF;  /* 더 밝은 파스텔 하늘색 */
  box-shadow: 0 0 12px #9EB2FF;
  color: #FFFFFF;  /* 하얀색 유지 */
}

/* 부드러운 테두리 반짝임 */
/* 공통 스타일 */
.glow-text {
  font-size: 40px;
  font-weight: bold;
  color: #4B5EAA; /* 파스텔 하늘색 진한 편 */
  text-align: center;
  /* 테두리 그림자 - 은은한 파스텔 블루톤 */
  text-shadow:
    0 0 4px rgba(117, 140, 246, 0.6),
    0 0 6px rgba(150, 170, 255, 0.4),
    0 0 8px rgba(183, 205, 255, 0.3);
  animation: borderGlow 3.5s ease-in-out infinite alternate;
}

/* 감정별 약간 색상 변형 */
/* 긍정 */
.positive.glow-text {
  text-shadow:
    0 0 5px rgba(135, 184, 255, 0.7),
    0 0 8px rgba(175, 210, 255, 0.5),
    0 0 12px rgba(210, 230, 255, 0.4);
}

/* 부정 */
.negative.glow-text {
  text-shadow:
    0 0 5px rgba(115, 135, 190, 0.7),
    0 0 8px rgba(150, 170, 210, 0.5),
    0 0 12px rgba(180, 200, 230, 0.4);
}

/* 중립 */
.neutral.glow-text {
  text-shadow:
    0 0 5px rgba(140, 160, 200, 0.7),
    0 0 8px rgba(170, 190, 230, 0.5),
    0 0 12px rgba(200, 220, 255, 0.4);
}

/* 테두리 부드럽게 반짝임 애니메이션 */
@keyframes borderGlow {
  0%, 100% {
    text-shadow:
      0 0 3px rgba(117, 140, 246, 0.5),
      0 0 5px rgba(150, 170, 255, 0.3),
      0 0 7px rgba(183, 205, 255, 0.2);
  }
  50% {
    text-shadow:
      0 0 7px rgba(117, 140, 246, 1),
      0 0 10px rgba(150, 170, 255, 0.8),
      0 0 15px rgba(183, 205, 255, 0.6);
  }
}
</style>
""", unsafe_allow_html=True)

# --- 메인 UI ---
st.markdown('<h1 style="text-align:center; color:#7083D9; margin-bottom: 0;">💡 한글 감정 분석 AI 🔍</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#5A75C9; margin-top: 0;">문장을 입력하면 감정을 분석해드려요! 😎</p>', unsafe_allow_html=True)

text = st.text_area("👇 감정을 알고 싶은 문장을 입력해 주세요:", height=150, placeholder="예) 오늘은 너무 행복해요! 🌞")

if st.button("✨ 감정 분석하기 ✨"):
    if text.strip() == "":
        st.warning("⚠️ 문장을 입력해 주세요!")
    else:
        result = model.predict([text])[0]

        # 감정별 CSS 클래스, 이모지, 배경색 설정
        style_map = {
            "긍정": ("positive glow-text", "😊💖🎈", "#E6EEFF"),
            "부정": ("negative glow-text", "😢💔🌧️", "#D4DAF8"),
            "중립": ("neutral glow-text", "😐📘🍃", "#E0E6F9")
        }
        css_class, emoji, bg_color = style_map.get(result, ("neutral glow-text", "🤔", "#F0F4FF"))

        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding:20px; border-radius:12px; text-align:center; margin-top: 20px;">
                <div class="{css_class}">
                    예측 감정: {result} {emoji}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("💡 *Streamlit과 Naive Bayes로 구현된 간단한 감정 분석기입니다.*")
