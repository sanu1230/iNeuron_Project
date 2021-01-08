from flask import Flask, render_template, request
import pickle
import datetime
import threading
import numpy as np
from keras.models import load_model
import pickle
import json
import numpy as np
import nltk
import threading
import pyttsx3 as pp
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from keras.models import load_model
import speech_recognition as sp
from datetime import datetime
import random
from time import sleep
import pymongo
import pymysql
import func
import eligibility
from datetime import timedelta

from flask import session, app




model = load_model('chatbot_model.h5')

app = Flask(__name__)

app.config['SECRET_KEY'] = '767989afdf678bnhlk'
app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=1)

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

##initializing variables
s = 0
q = 0

global lastReplyTime, user
user = []
bot = []
response = ''
customerId = 0
StartTime = 0
c = datetime.now()
lastReplyTime = (c.hour * 60 * 60) + (c.minute * 60) + c.second  # lastreplytime is set on current time
print("No_Conversation: ", lastReplyTime)

@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=1)

@app.route('/')
def man():
    session.permanent = True
    intro=[]
    intro.append("Bot : Hi, I am AssistBot. Your customer service agent. How may I help you?")
    return render_template('home.html',user_input=intro)

def r():  ##takes user inputs and bot outputs and insert into a array to later send to html file
    print("Inside chatbot_response")
    global lastReplyTime
    def replytime():
        global lastReplyTime, user, bot
        # Session Recording
        # Resetting lastReplyTime
        c = datetime.now()
        current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
        lastReplyTime = current_time
        print("chat_response: ", lastReplyTime)
        print(user)
        print(bot)
        return lastReplyTime

    user_input = request.form["user_input"]
    lastReplyTime = replytime()
    insert_sql(user_input)
    user = user_list()
    print("user", user)
    return (user)




def date_time():  ##getting date and time
    currentDT = datetime.datetime.now()
    return (str(currentDT)[0:19])

def insert_sql(user_input):  ##inserting user inputs, bot outputs and time into database
    global s
    current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
    s = s + 1
    resp = str(enter_proper_response(user_input))
    try:
        record = {'UserId': 123,
                  'UserInput': user_input,
                  'BotInput':resp,
                  'InputTime' : current_time
                 }
        collection.insert_one(record)
    except:
        print("Some error in the tables, check if table does exist and its inputs")

def user_list():  ##extracting user inputs from user_bot_chat database
    user = []
    val = { 'UserId': 123 }
    sql = collection.find(val)
    print("sql",sql)
    r = []
    r.append("Bot : Hi, I am AssistBot. Your customer service agent. How may I help you?")
    for document in sql:
        r.append("You : "+ str(document['UserInput']))
        r.append("Bot : "+document['BotInput'])
    return r

@app.route('/process', methods=['POST'])
def process():  ##called when user input is given and submit button is pressed
    return render_template("index.html", user_input=r())


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    print("res predict : ",res)
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    print("results : ",results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print("return_list : ",return_list)
    return return_list

def getResponse(query, ints, userID, intents, user):
    print("Inside getResponse")
    tag = func.getTag(query, ints, userID)
    print("tag ",tag)
    response = func.contextResponse(query, tag, userID, intents, user)
    return response

def push_to_mongodb(uniqueSessionId):
    global StartTime, customerId, user, bot
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = "Metadata"
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    dataBase = client[DB_NAME]
    COLLECTION_NAME = "Session_Info"
    collectionSession = dataBase[COLLECTION_NAME]
    record = {'SessionId': uniqueSessionId,
              'StartTime': StartTime,
              'EndTime': datetime.now(),
              'CustomerId': customerId,
              'CustomerStatus': 'Active',
              'DataFileName': 'abc.txt'}
    collectionSession.insert_one(record)
    print("Record 1", record)
    COLLECTION_NAME = "Session_Data"
    collectionSession = dataBase[COLLECTION_NAME]
    record2 = {'SessionId': uniqueSessionId,
               'CustomerId': customerId,
               'UserConv': user,
               'BotResp': bot,
               }
    collection.insert_one(record2)
    print("Record 2", record2)

def enter_proper_response(query):
    print("Inside enter_proper_response")
    print("Query ",query)
    global response, bot, user, intents
    userID = '123'
    ints = predict_class(query, model)
    print("ints :",ints)
    print("ints[0] :", ints[0])
    if ints[0]['intent'] == "goodbye":  # or "anything_else_no"
       tag = "goodbye"
       response = func.contextResponse(query, tag, userID, intents, user)
       return response
    else:
       print("Inside else enter_proper_response")
       tag = ints[0]['intent']
       print("tag :", tag)
       #response = func.contextResponse(query, tag, userID, intents, user)
       response = getResponse(query, ints, userID, intents, user)
       print("response",response)
       if response == "calc_eli":
           print("Cal eli")
           response = eligibility.eligibility_response(user)
           return response
       else:
           response = response
           return response
       return response
    return ints


if __name__ == "__main__":
    global collection
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = "Metadata"
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    dataBase = client[DB_NAME]
    COLLECTION_NAME = "User_Convo"
    collection = dataBase[COLLECTION_NAME]
    app.run(debug=True)
    #try:  ##connects to the database
    #    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    #    DB_NAME = "Metadata"
    #    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    #    dataBase = client[DB_NAME]
    #    COLLECTION_NAME = "User_Convo"
    #    collection = dataBase[COLLECTION_NAME]
    #except:
    #    print("connection error - hostname or password incorrect")
    #app.run(host='127.10.0.0', port=int('8000'), debug=True)  ##0.0.0.0.,80














