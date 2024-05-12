import requests
import joblib
import string
import re
import nltk
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu

nltk.download('stopwords') #stopwords removal
nltk.download('punkt') #tokenization

st.set_page_config(page_title='Sentiment review', page_icon='📊')


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


loaded_model = joblib.load(open('sentiment_model', 'rb'))


st.header("Sentiment Analysis")
with st.sidebar:
    choose = option_menu(None, ["About", "Contact us"],
                         icons=['house','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#F5B7B1"},
    }
    )
    if choose == "About":
        st.write("""This project is about predicting whether
                    the text is either postive, negative or 
                    neutral using NLP and other tools that helps
                    in developing this project""")
    elif choose == "Contact us":
        st.write("""
                    Julia Joseph
                 
                    2022170117@cis.asu.edu.eg
                 
                    Noureen Mohamed
                 
                    2022170475@cis.asu.edu.eg""")
        

lottie_link = "https://lottie.host/927e04be-3392-4082-a7cd-7f53bf8485ec/VYKl2qIKol.json"
animation = load_lottie(lottie_link)

st.write('-------')
st.write("Enter your comment to predict sentiment status")

with st.container():
    left, right = st.columns(2)
    with left:
        Name=st.text_input("Name: ")
        text = st.text_input("Review: ")
        def text_editor(text):
            # Punctuation removal
            punctuation_chars = set(string.punctuation)
            text_without_punctuation = ''.join(char for char in text if char not in punctuation_chars)
            
            # Removing emojis
            emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
            text_without_emojis = emoji_pattern.sub(r'', text_without_punctuation)
            
            # Remove stop words
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text_without_emojis)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            filtered_sentence = ' '.join(filtered_words)
            
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(filtered_sentence)
            lemmatized_sentence = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
            
            tfidf_vectorizer = TfidfVectorizer(max_features=2467)  
            # Set max_features to desired size
            tfidf_sentence = tfidf_vectorizer.fit_transform([lemmatized_sentence])
            tfidf_array = tfidf_sentence.toarray()
            if tfidf_array.shape[1] < 2467:
                padding = np.zeros((1, 2467 - tfidf_array.shape[1]))
                tfidf_array = np.hstack((tfidf_array, padding))


            return tfidf_array

    with right:
        st_lottie(animation, speed=1, height=350, key='initial')
  

    if st.button("Predict"):
        if text:
            st.write("Hi "+ Name +"!")
            preprocessed_text = text_editor(text)
            pred_y = loaded_model.predict(preprocessed_text)
            if pred_y == 2:
                st.success("Your comment is positive")
                st.balloons()
            elif pred_y == 0:
                st.error("Your comment is negative")
            elif pred_y == 1:
                st.info("Your comment is neutral")

            