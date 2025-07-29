import streamlit as st 
import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
df=pd.read_csv("korean_sentiment2.csv")
model=make_pipeline(CountVectorizer(),MultinomialNB())
model.fit(df['text'],df['label'])
st.title("감정 분석 AI(한글)")
text=st.text_area("문장을 입력해 보세요")
if st.button("분석하기"):
    result=model.predict([text])[0]
    st.write(f"예측 감정:**{result}**")