import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# import pandas as pd
# import pickle
# import numpy as np
# import datetime
# from time import sleep
# import time
# import sys, os, signal

# import threading
import pymongo
from func import *
from eligibility import *
from model import *
from pl_apply import *
import requests
from flask import Flask, render_template, jsonify, request
import json



data_file = open('data/intents.json').read()
intents = json.loads(data_file)

data_file1 = open('data/faq_data.json').read()
faqs = json.loads(data_file1)

for intent in faqs['intents']:
    intents['intents'].append(intent)

# train_x, train_y = training_data(intents)
# model_creation(train_x, train_y)

story_conv_file = open('data/story_conv.json').read()
story_conv = json.loads(story_conv_file)

app = Flask(__name__)

global result, voice_query, lastReplyTime, c, userID

c = datetime.now()

##initializing variables
s = 0
q = 0
Fc = 0
response = ''
customerId = 0
StartTime = 0
user = []
bot = []

result = []

# --- User Interface ---
@app.route('/')
def home():
    global userID
    userID = str(random.randint(1, 198))
    result.append([('',''),("Bot",  "Hi, I am AssistBot. Your customer service agent. How may I help you?")])
    mike_status="no"
    resp = "Hi, I am AssistBot. Your customer service agent. How may I help you?"
    killSession="no"
    return render_template('index.html', user_input=result, mike_status=mike_status, botResp=resp, userID=userID, killSession=killSession)


# @app.route("/predict", methods=['POST'])
# @cross_origin()
# def inference():
#     if request.method == 'POST':
#         json_data = request.get_json()
#
#         result = predict(input_data=json_data)
#
#         return jsonify(result)


@app.route('/process', methods=['POST'])
def process():  ##called when user input is given and submit button is pressed
    global userID, user
    print("Process Called")
    query = request.form["user_input"]
    print("user_input : ", query)
    user.append(query)
    if query == "TimeOut":
        resp = bot_insert_sql(query)
    else:
        resp = insert_sql(query)
        bot.append(resp)
        print("Bot resp3 : ", resp)

    result.append([("You", query), ("Bot", resp)])
    mike_status = "yes" if request.form["mic_status"] == "on" else "no"
    print("mike_status : ",mike_status,request.form["mic_status"])
    userID = request.form["userID"]
    killSession = request.form["killSession"]
    print("killSession : ",killSession)
    # conv_list = user_list()
    if killSession == "yes":
        val = {'UserId': userID}
        collection.remove(val)
        pushconv_to_mongodb(userID, result)
    return render_template("index.html", user_input=result, mike_status=mike_status,botResp=resp,userID=userID,killSession=killSession)



def bot_insert_sql(user_input):  ##inserting user inputs, bot outputs and time into database
    global s, userID, c
    current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
    s = s + 1
    resp = "Goodbye"
    try:
        record = {'UserId': userID,
                  'UserInput': " ",
                  'BotInput':resp,
                  'InputTime' : current_time
                 }
        collection.insert_one(record)
    except:
        print("Some error in the tables, check if table does exist and its inputs")
    return resp


def insert_sql(query):  ##inserting user inputs, bot outputs and time into database
    global s, userID, c
    current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
    s = s + 1
    resp = enter_proper_response(query)
    # print("Bot_resp2: ", resp)
    try:
        record = {'UserId': userID,
                  'UserInput': query,
                  'BotInput': resp,
                  'InputTime' : current_time
                 }
        collection.insert_one(record)
    except:
        print("Some error in the tables, check if table does exist and its inputs")
    return resp


def user_list():  ##extracting user inputs from user_bot_chat database
    global userID
    val = { 'UserId': userID }
    sql = collection.find(val)
    print("sql", sql)
    for document in sql:
        result.append([("You", document['UserInput']), ("Bot",  document['BotInput'])])
    return result


def getResponse(query, ints, intents, cTag):
    tag = getTag(query, ints, cTag)
    response = contextResponse(query, tag, intents, cTag)
    return response


def enter_proper_response(query):
    global response, bot, user, intents, userID
    cTag = 'context'
    if query != '':
        ints = predict_class(query, words, model)
        response = getResponse(query, ints, intents, cTag)
        if response == "calc_eli":
            response = eligibility_response(user)
        elif response == 'apply_pl':
            response = apply_pl(user)
        else:
            response = response
    else:
        response = "Please text or say your query. I will be glad to help you."
        return response
    # print("Bot_resp1 : ", response)
    return response


# --- Write session Convo to database
def pushconv_to_mongodb(userID, rlist):
    global StartTime, customerId, user, bot
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = "Metadata"
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    dataBase = client[DB_NAME]
    COLLECTION_NAME = "Session_Info"
    collection = dataBase[COLLECTION_NAME]
    record = {'SessionId': userID,
              'StartTime': StartTime,
              'EndTime': datetime.now(),
              'CustomerId': userID,
              'CustomerStatus': 'Active',
              'Convo': rlist}
    collection.insert_one(record)


#    --- Keeping track of chat history ---
# def date_time():  ##getting date and time
#     currentDT = datetime.datetime.now()
#     return (str(currentDT)[0:19])


if __name__ == "__main__":
    global collection
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = "Metadata"
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    dataBase = client[DB_NAME]
    COLLECTION_NAME = "User_Convo"
    collection = dataBase[COLLECTION_NAME]
    app.run(threaded=True, debug=True, use_reloader= False)
