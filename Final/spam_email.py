import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score, ConfusionMatrixDisplay, classification_report
def welcome_page():
    st.title("Spam Email Detection")
    st.image("https://www.pantechelearning.com/wp-content/uploads/2021/12/Spam-classification.png")
    text1 = """
    #### Are you mad that your email program always throw important email into spam folder?
    #### Why not try this one?
    """
    st.markdown(text1)
    if st.button("Enter"):
        st.session_state.page = "main"
def main_page():
    button2 = st.sidebar.button("See Description")
    if button2:
        st.session_state.page = "welcome"
        st.experimental_rerun()
    if not button2:
        st.title("Spam Email Detection")
        st.sidebar.title("Design your methods")
        text2 = """
        ### Introduction
        Email is widely used now, but also with the rapid development of spam email. The email containing unsolicited and potentially harmful content is bothering every email users. Although email providers have developed many spam email filters, there are still high chance that the spam email is not recognized or normal email is thrown into the spam folder.
        We will face this problem directly. I have applied 2007 TREC Public Spam Corpus and Enron-Spam Dataset into machine learning, to help detect the spam email. Over 83446 records of email were put into the ML methods which are labelled as either spam or not-spam.
        ### How to use it?
        Enter the text version of the email you received, and it will automatically identify whether it is spam or not. You can also check the confusion matrix to see the accuracy of our program 
        """
        st.markdown(text2)
        df = pd.read_csv("combined_data.csv")
        a = df.iloc[4,1]
        b = df.iloc[1,1]
        method = st.sidebar.radio("Choose the method",["Dataset Information","Run Detector"])
        pos = df[df['label'] == 1].sample(5000)
        neg = df[df['label'] == 0].sample(5000)
        df2 = pd.concat([pos,neg],axis=0)
        X = df2['text']
        y = df2['label']
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        if method == "Dataset Information": 
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.histplot(df,x='label')
            ax.set_title("distribution of email type (0: normal; 1: spam)")
            st.pyplot(fig)
            st.markdown("#### Example of normal email")
            st.markdown(a)
            st.markdown("#### Example of spam email")
            st.markdown(b)
            t1 = ConfusionMatrixDisplay(cm)
            t1.plot()
            st.pyplot()
        elif method == "Run Detector":
            user_input = st.text_input("Enter your email here:", "") 
            press = st.button("Detect")
            if press:
                inp= vectorizer.fit_transform([user_input]).toarray()
                y_p = model.predict(inp)
                st.write("Email Content")
                st.write(user_input)
                if y_p[0] == 0:
                    st.write("This is a normal email")
                elif y_p[0] == 1:
                    st.write("This is a spam email")
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "main":
    main_page()

    
