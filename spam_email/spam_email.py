import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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
    button2 = st.sidebar.button("Return to welcome page")
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
        df = pd.read_csv("combined_data.csv",index_col=0)
        method = st.sidebar.radio("Choose the method",["Dataset Information","Machine Learning Overview","Run Detector"])
        pos = df[df['label'] == 1].sample(2000)
        neg = df[df['label'] == 0].sample(2000)
        df2 = pd.concat([pos,neg],axis=0)
        X = df2['text']
        y = df2['label']
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        names = [
            "Nearest Neighbors",
            "Logistic Regression (Recommended)",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
            "QDA",
        ]
        classifiers = [
        KNeighborsClassifier(2),
        LogisticRegression(),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        ]
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if method == "Dataset Information": 
            st.markdown(text2)
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.histplot(df,x='label')
            ax.set_title("distribution of email type (0: normal; 1: spam)")
            st.pyplot(fig)
            em = st.radio("#### Show me",["A normal email","A spam email"])
            if em == "A normal email":
                a = np.random.randint(neg.shape[0])
                st.markdown(neg.iloc[a,1])
            elif em == "A spam email":
                b = np.random.randint(pos.shape[0])
                st.markdown(neg.iloc[b,1])
            
        elif method == "Machine Learning Overview":
            method = st.selectbox("Choose Classifier",names)
            i = names.index(method)
            clf = classifiers[i]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            predicted = clf.predict(X_test)
            st.markdown(f"#### Accuracy = {score}")
            disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
            disp.figure_.suptitle(f"Confusion Matrix for {method}")
            st.pyplot()
        elif method == "Run Detector":
            user_input = st.text_input("Enter your email here:", "") 
            method = st.selectbox("Choose Classifier",names)  
            press = st.button("Detect")
            if press:
                df2.iloc[0,1] = user_input
                X = df2['text']
                y = df2['label']
                X = vectorizer.fit_transform(X).toarray()
                inp = X[[0],:]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                i = names.index(method)
                clf = classifiers[i]             
                clf.fit(X_train, y_train)
                y_p = clf.predict(inp)
                st.write("#### Email Content")
                st.write(user_input)
                if y_p[0] == 0:
                    st.write("#### This is a normal email")
                elif y_p[0] == 1:
                    st.write("#### This is a spam email")
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "main":
    main_page()

    
