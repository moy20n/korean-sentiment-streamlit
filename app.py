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
st.set_page_config(page_title="감정 분석 AI", page_icon="🧠", layout="centered")

# --- 사이드바 ---
st.sidebar.title("✨ 감정 분석기 Ver. ChatGPT ✨")
st.sidebar.markdown("한글 문장을 입력하면 감정을 분석해드려요! 😊\n\nMade with ❤️ by 호연")

# --- CSS 스타일 + 애니메이션 ---
st.markdown("""
<style>
/* 기본 배경 */
.main {
  background-color: #FFF8F0;
  font-family: 'Pretendard', sans-serif;
}

/* 텍스트 영역 배경 */
.stTextArea > div > textarea {
  background-color: #F0F8FF;
  font-size: 16px;
  border-radius: 10px;
  padding: 10px;
}

/* 버튼 꾸미기 */
div.stButton > button:first-child {
  background-color: #87CEFA;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.6em 1.2em;
  font-size: 18px;
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.3s ease;
}
div.stButton > button:first-child:hover {
  background-color: #00BFFF;
  color: white;
}

/* 빛나는 네온 효과 */
/* 긍정 */
.positive {
  font-size: 40px;
  font-weight: bold;
  color: #39ff14;
  text-align: center;
  text-shadow:
    0 0 5px #39ff14,
    0 0 10px #39ff14,
    0 0 20px #39ff14,
    0 0 40px #0fa,
    0 0 80px #0fa,
    0 0 90px #0fa,
    0 0 100px #0fa,
    0 0 150px #0fa;
  animation: flickerGreen 1.5s infinite alternate, shake 0.5s infinite;
}

/* 부정 */
.negative {
  font-size: 40px;
  font-weight: bold;
  color: #ff073a;
  text-align: center;
  text-shadow:
    0 0 5px #ff073a,
    0 0 10px #ff073a,
    0 0 20px #ff073a,
    0 0 40px #ff073a,
    0 0 80px #ff073a;
  animation: flickerRed 1.5s infinite alternate, shakeStrong 0.3s infinite;
}

/* 중립 */
.neutral {
  font-size: 40px;
  font-weight: bold;
  color: #1e90ff;
  text-align: center;
  text-shadow:
    0 0 5px #1e90ff,
    0 0 10px #1e90ff,
    0 0 20px #1e90ff;
  animation: flickerBlue 2s infinite alternate, shake 1.5s infinite;
}

/* 애니메이션 정의 */
@keyframes flickerGreen {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
@keyframes flickerRed {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
@keyframes flickerBlue {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-5px) rotate(-2deg);}
  50% { transform: translateX(5px) rotate(2deg);}
  75% { transform: translateX(-5px) rotate(-2deg);}
}

@keyframes shakeStrong {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-10px) rotate(-5deg);}
  50% { transform: translateX(10px) rotate(5deg);}
  75% { transform: translateX(-10px) rotate(-5deg);}
}
</style>
""", unsafe_allow_html=True)

# --- 메인 UI ---
st.markdown('<h1 style="text-align:center; color:#2E86C1; margin-bottom: 0;">💡 한글 감정 분석 AI 🔍</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#555; margin-top: 0;">문장을 입력하면 감정을 분석해드려요! 😎</p>', unsafe_allow_html=True)

text = st.text_area("👇 감정을 알고 싶은 문장을 입력해 주세요:", height=150, placeholder="예) 오늘은 너무 행복해요! 🌞")

if st.button("✨ 감정 분석하기 ✨"):
    if text.strip() == "":
        st.warning("⚠️ 문장을 입력해 주세요!")
    else:
        result = model.predict([text])[0]

        # 감정별 클래스, 이모지, 배경색 설정
        style_map = {
            "긍정": ("positive", "😊💖🎈", "#d0f0c0"),
            "부정": ("negative", "😢💔🌧️", "#fcdede"),
            "중립": ("neutral", "😐📘🍃", "#e0e0e0")
        }
        css_class, emoji, bg_color = style_map.get(result, ("neutral", "🤔", "#f0f0f0"))

        # 결과 출력
        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding:20px; border-radius:12px; text-align:center; margin-top: 20px;">
                <div class="{css_class}">
                    예측 감정: {result} {emoji}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 풍선 효과
        st.balloons()

st.markdown("---")
st.markdown("💡 *Streamlit과 Naive Bayes로 구현된 간단한 감정 분석기입니다.*")
