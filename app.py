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

# --- CSS 스타일 + 하늘색 테마 + 부드러운 테두리 반짝임 ---
st.markdown("""
<style>
/* 기본 배경 */
.main {
  background-color: #F0F4FF;  /* 연한 하늘색 느낌 */
  font-family: 'Pretendard', sans-serif;
}

/* 텍스트 영역 배경 */
.stTextArea > div > textarea {
  background-color: #BFBFFF;  /* 은은한 진한 하늘색 */
  font-size: 16px;
  border-radius: 10px;
  padding: 10px;
  color: #000099;
  font-weight: 600;
  border: 1.5px solid #6666CC;
}

/* 버튼 꾸미기 */
div.stButton > button:first-child {
  background-color: #0000CC;  /* 진한 하늘색 */
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.6em 1.2em;
  font-size: 18px;
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.3s ease;
  box-shadow: 0 0 8px #3333CC;
}
div.stButton > button:first-child:hover {
  background-color: #4040FF;  /* 밝고 진한 하늘색 */
  box-shadow: 0 0 14px #6666CC;
  color: white;
}

/* 부드러운 테두리 반짝임 */
/* 공통 스타일 */
.glow-text {
  font-size: 40px;
  font-weight: bold;
  color: #000099; /* 진한 하늘색 글씨 */
  text-align: center;
  /* 테두리 그림자 - 은은한 하늘색 계열 */
  text-shadow:
    0 0 4px rgba(51, 51, 204, 0.8),   /* #3333CC */
    0 0 6px rgba(102, 102, 204, 0.6), /* #6666CC */
    0 0 8px rgba(153, 153, 204, 0.4); /* #9999CC */
  animation: borderGlow 3.5s ease-in-out infinite alternate;
}

/* 감정별 약간 색상 변형 */
/* 긍정 */
.positive.glow-text {
  text-shadow:
    0 0 5px rgba(0, 128, 255, 0.8),  /* #0080FF 약간 밝은 하늘색 */
    0 0 8px rgba(0, 153, 255, 0.6),
    0 0 12px rgba(51, 204, 255, 0.4);
}

/* 부정 */
.negative.glow-text {
  text-shadow:
    0 0 5px rgba(0, 51, 102, 0.8),   /* #003366 더 어두운 하늘색 */
    0 0 8px rgba(0, 76, 153, 0.6),
    0 0 12px rgba(0, 102, 204, 0.4);
}

/* 중립 */
.neutral.glow-text {
  text-shadow:
    0 0 5px rgba(51, 102, 153, 0.8),  /* #336699 중간톤 하늘색 */
    0 0 8px rgba(77, 124, 153, 0.6),
    0 0 12px rgba(115, 115, 153, 0.4);
}

/* 테두리 부드럽게 반짝임 애니메이션 */
@keyframes borderGlow {
  0%, 100% {
    text-shadow:
      0 0 3px rgba(51, 51, 204, 0.6),
      0 0 5px rgba(102, 102, 204, 0.4),
      0 0 7px rgba(153, 153, 204, 0.2);
  }
  50% {
    text-shadow:
      0 0 7px rgba(51, 51, 204, 1),
      0 0 10px rgba(102, 102, 204, 0.8),
      0 0 15px rgba(153, 153, 204, 0.6);
  }
}
</style>
""", unsafe_allow_html=True)

# --- 메인 UI ---
st.markdown('<h1 style="text-align:center; color:#0000CC; margin-bottom: 0;">💡 한글 감정 분석 AI 🔍</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#3333CC; margin-top: 0;">문장을 입력하면 감정을 분석해드려요! 😎</p>', unsafe_allow_html=True)

text = st.text_area("👇 감정을 알고 싶은 문장을 입력해 주세요:", height=150, placeholder="예) 오늘은 너무 행복해요! 🌞")

if st.button("✨ 감정 분석하기 ✨"):
    if text.strip() == "":
        st.warning("⚠️ 문장을 입력해 주세요!")
    else:
        result = model.predict([text])[0]

        # 감정별 CSS 클래스, 이모지, 배경색 설정
        style_map = {
            "긍정": ("positive glow-text", "😊💖🎈", "#BFDFFF"),
            "부정": ("negative glow-text", "😢💔🌧️", "#AAB8FF"),
            "중립": ("neutral glow-text", "😐📘🍃", "#D1D9FF")
        }
        css_class, emoji, bg_color = style_map.get(result, ("neutral glow-text", "🤔", "#E0E8FF"))

        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding:20px; border-radius:12px; text-align:center; margin-top: 20px;">
                <div class="{css_class}">
                    예측 감정: {result} {emoji}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.balloons()

st.markdown("---")
st.markdown("💡 *Streamlit과 Naive Bayes로 구현된 간단한 감정 분석기입니다.*")
