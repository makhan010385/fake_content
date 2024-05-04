import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests
import urllib.request
import time
#Importing all required library
import nltk
import re
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import seaborn as sns 
import time
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
st.title("Web  Content Scrapper ")
authors = []
dates = []
statements = []
sources = []
targets = []
 
def scrape_website(page_number):
    page_num = str(page_number)
    URL= st.text_input('Enter the URL')
#   URL = 'https://www.politifact.com/factchecks/list/?page='+page_num
    webpage = requests.get(URL)
    soup = BeautifulSoup(webpage.text, 'html.parser')
    statement_footer = soup.find_all ('footer' , attrs={'class' : 'm-statement__footer'})
    statement_quote = soup.find_all ('div' , attrs={'class' : 'm-statement__quote'})
    statement_meta = soup.find_all ('div' , attrs={'class' : 'm-statement__meta'})
    target = soup.find_all ('div' , attrs={'class' : 'm-statement__meter'})

    for i in statement_footer:
        link1 = i.text.strip()
        name_and_date= link1.split()
        first_name = name_and_date[1]
        last_name = name_and_date[2]
        full_name = first_name + ' ' + last_name
        month = name_and_date[4]
        day = name_and_date[5]
        year = name_and_date[6]
        date = month+' '+day+' '+year
        dates.append(date)
        authors.append(full_name)   

    for i in statement_quote:
        link2 = i.find_all('a')
        statement_text = link2[0].text.strip()
        statements.append(statement_text)

    for i in statement_meta:
        link3 = i.find_all('a')
        source_text = link3[0].text.strip()
        sources.append(source_text)

    for i in target:
        link4 = i.find('div' , attrs={'class' : 'c-image'}).find('img').get('alt')
        targets.append(link4)   
n = 2
for i in range(1, n):
    scrape_website(i)
data = pd.DataFrame(columns = ['author','statement','source','date','target'])
data['author']= authors
data['statement']= statements
data['source']= sources
data['date']= dates
data['target']= targets     

def  getBinaryNumTarget(target):
    if target == 'mostly-true':
        return 1
    elif target=='half-true':
        return 1
    elif target =='true':
        return 1
    else:
       return 0

def  getBinaryTarget(target):
    if target== 'mostly-true' :
        return 'REAL'
    elif target=='half-true':
        return 'REAL'
    elif target =='true':
        return'REAL'
    else:
       return 'FAKE'
data['BinaryTarget']= data['target'].apply(getBinaryTarget)
data['BinaryNumTarget']= data['target'].apply(getBinaryNumTarget)     
if st.button('Show Data'):
    st.write(data)
else:
    st.write('Data us not availabe')
df = pd.read_csv("merged_files3.csv",encoding = "ISO-8859-1")

data1= df.fillna(' ')
if st.checkbox('Show dataframe'):
    st.write(data1)
st.subheader('Scatter plot')
species = st.multiselect('Show content per true& false?', data1['BinaryTarget'].unique())
col1 = st.selectbox('Which feature on x?', data1.columns[0:5])
col2 = st.selectbox('Which feature on y?', data1.columns[0:5])
new_df = data1[(data1['BinaryTarget'].isin(species))]
st.write(new_df)

def fake_Content_Detection_Using_Naive_Bayes():
    user = st.text_area("Enter Any News Headline: ")
    if len(user) < 1:
        st.write(" ")
    else:
        data1= df.fillna(' ')
        x = np.array(data1["statement"])
        y = np.array(data1["BinaryTarget"])
        cv = CountVectorizer()
        x = cv.fit_transform(x)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(xtrain, ytrain)
        st.subheader("Accuracy of all models ")
        x = np.array(data1["statement"])
        y = np.array(data1["BinaryTarget"])
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
fake_Content_Detection_Using_Naive_Bayes()


# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2, color='BinaryTarget')
# Plot!
st.plotly_chart(fig)
st.subheader('Histogram')
feature = st.selectbox('Which feature?', df.columns[0:5])
# Filter dataframe
new_df2 = df[(df['BinaryTarget'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color="BinaryTarget", marginal="rug")
st.plotly_chart(fig2)
st.subheader('Machine Learning models')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
import itertools
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

features= df[['author', 'statement', 'source', 'date']].values
labels = df['BinaryTarget'].values
x = np.array(data1["statement"])
y = np.array(data1["BinaryTarget"])
xtrain,xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7, random_state=1)
alg = ['Decision Tree', 'Support Vector Machine','Naive Bayes classifier','Logistic regression classifier','Random forest']
classifier = st.selectbox('Which algorithm?', alg)

def naive():
    start = time.time() 
    st.subheader("Naive Bayes classifier")
    dct = dict()
    NB_classifier = MultinomialNB()
    pipe1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('model', NB_classifier)])
    model1 = pipe1.fit(xtrain,ytrain)
    prediction1 = model1.predict(xtest)
    dct['Naive Bayes'] = round(accuracy_score(ytest, prediction1)*100,2)
    cm = metrics.confusion_matrix(ytest,prediction1)
    # st.write(plot_confusion_matrix(cm, classes=['Fake', 'Real']))
    st.write('Naive Bayes Confusion matrix: ', cm)
    stop = time.time()
    st.write(f"Training time Naive: {stop - start}s")
    return  st.write("accuracy: {}%".format(round(accuracy_score(ytest, prediction1)*100,2))) 
    
 
def LRC():
    start = time.time() 
    st.subheader("Logistic regression classifier")
    dct = dict()
    # Vectorizing and applying TF-IDF
    from sklearn.linear_model import LogisticRegression
    pipe2 = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

    # Fitting the model
    model2 = pipe2.fit(xtrain,ytrain)
    # Accuracy
    prediction2 = model2.predict(xtest)
    dct['Logistic Regression'] = round(accuracy_score(ytest, prediction2)*100,2)
    cm = metrics.confusion_matrix(ytest,prediction2)
    st.write('Naive Bayes Confusion matrix: ', cm)
    stop = time.time()
    st.write(f"Training time Logistic: {stop - start}s")
    return st.write("accuracy: {}%".format(round(accuracy_score(ytest, prediction2)*100,2)))   

def DTC():
    start = time.time() 
    st.subheader("Decision Tree classifier")
    dct = dict()
    from sklearn.tree import DecisionTreeClassifier
    # Vectorizing and applying TF-IDF
    pipe3 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('model', DecisionTreeClassifier(criterion= 'entropy',max_depth = 20, splitter='best',random_state=42))])
    # Fitting the model
    model3 = pipe3.fit(xtrain, ytrain)
    # Accuracy
    prediction3 = model3.predict(xtest)
    dct['Decision Tree'] = round(accuracy_score(ytest, prediction3)*100,2)
    cm = metrics.confusion_matrix(ytest, prediction3)
    # st.write(plot_confusion_matrix(cm, classes=['Fake', 'Real']))
    st.write(' Decesion Tree Confusion matrix: ', cm)
    stop = time.time()
    st.write(f"Training time Decision Tree: {stop - start}s")
    return  st.write("accuracy: {}%".format(round(accuracy_score(ytest, prediction3)*100,2)))
  
def RFC():
    start = time.time() 
    st.subheader('Random Forest classifier')
    dct = dict()
    from sklearn.ensemble import RandomForestClassifier 
    pipe4 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('model', RandomForestClassifier(n_estimators=10, criterion="entropy"))])
    model4 = pipe4.fit(xtrain, ytrain)
    prediction4 = model4.predict(xtest)
    dct['Random Forest'] = round(accuracy_score(ytest, prediction4)*100,2)
    accuracy = accuracy_score(ytest,prediction4)
    st.write(accuracy)
    cm = metrics.confusion_matrix(ytest, prediction4)
    # st.write(plot_confusion_matrix(cm, classes=['Fake', 'Real']))
    st.write('Random Forest Confusion matrix: ', cm)
    stop = time.time()
    st.write(f"Training time Random Forest classifier: {stop - start}s")
    return st.write("accuracy: {}%".format(round(accuracy_score(ytest, prediction4)*100,2)))
def SVM():
    start = time.time()
    st.subheader("SVM classifier")
    dct = dict()
    from sklearn import svm
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel
    pipe5 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('model', clf)])
    model5 = pipe5.fit(xtrain, ytrain)
    prediction5 = model5.predict(xtest)
    dct['SVM'] = round(accuracy_score(ytest, prediction5)*100,2)
    cm = metrics.confusion_matrix(ytest, prediction5)
    # plot_confusion_matrix(cm, classes=['Fake', 'Real'])
    st.write('Support  Vector Machine Confusion matrix: ', cm)
    stop = time.time()
    st.write(f"Training time Naive: {stop - start}s")
    return st.write("accuracy: {}%".format(round(accuracy_score(ytest, prediction5)*100,2)))

if classifier=='Naive Bayes classifier':
    naive()
   
elif classifier == 'Support Vector Machine':
    SVM()

elif classifier == 'Decision Tree':
    DTC()

elif classifier == 'Random forest':
    RFC()
else :
    LRC()





