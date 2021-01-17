import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

def encode(x):
    if x == 'not happy':
        return 0
    else:
        return 1
def encoding(data):
    data.Is_Response = data.Is_Response.apply(encode)
    return data


#Data Cleaning
def removeApostrophe(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"can\'t", "can not", review)
    phrase = re.sub(r"n\'t", " not", review)
    phrase = re.sub(r"\'re", " are", review)
    phrase = re.sub(r"\'s", " is", review)
    phrase = re.sub(r"\'d", " would", review)
    phrase = re.sub(r"\'ll", " will", review)
    phrase = re.sub(r"\'t", " not", review)
    phrase = re.sub(r"\'ve", " have", review)
    phrase = re.sub(r"\'m", " am", review)
    return phrase
def cleaning(df):
    all_reviews = list()
    lines = df["Description"].values.tolist()
    for text in lines:
        text = text.lower() # converting the text to lower case
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text) # removes URL'S
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text) #removes punctuation
        text = removeApostrophe(text)
        tokens = word_tokenize(text) #tokenizing
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()] #filtering only text data
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not") #removing "not" from stopwords as it is sentimental analysis
        PS = PorterStemmer()
        words = [PS.stem(w) for w in words if not w in stop_words] #stemming and removing stopwords
        words = ' '.join(words) #joining strings
        all_reviews.append(words)
    return all_reviews


def building_model(X, y):
    global X_train, X_test, y_train, y_test
    global predictions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tvec = TfidfVectorizer()
    svc = SVC(C=5, gamma=0.1, probability=True, decision_function_shape='ovr')
    model = Pipeline([('vectorizer', tvec), ('classifier', svc)])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    filename = 'model_hotelreviews.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print("model saved")

def metrics(predictions, y_test):
    print("Accuracy : ", accuracy_score(predictions, y_test))
    print("Precision : ", precision_score(predictions, y_test, average='weighted'))
    print("Recall : ", recall_score(predictions, y_test, average='weighted'))
def predict(model,review):
    l = []
    l.append(review)
    sent = model.predict(l)
    if sent[0]== 0:
        return "Negative"
    else:
        return "Positive"

#############################
data = pd.read_csv('hotel_reviews.csv')[:5000]
data = encoding(data)
data['cleaned_reviews'] = cleaning(data)
X = data['cleaned_reviews']
y = data["Is_Response"]
building_model(X, y)
metrics(predictions, y_test)


#-----------------------------------------------------------------------
def predict(review):
    model = pickle.load(open('model_hotelreviews.pkl', 'rb'))
    l = []
    l.append(review)
    sent = model.predict(l)
    if sent[0]== 0:
        return "Negative"
    else:
        return "Positive"
customer_reviews = ['Siddharth from MyLoanCare helped well and the loan got processed fast','Its a very good experience with you. Here I found the best deal and best interest rate of all banks'
                   ,'Very good. Special thanks to Saurabh Gandhi from MyLoanCare and Ram from HDFC Bank',
                   'Amazing service. Really happy with the team. Definitely recommend it to everyone.',
                   'Worst service by the bank',
                   'had very bad expierience']
for i in customer_reviews:
    print(predict(i))
    
    
