import layout1
import voice1
import chatbot1
import pymongo
import threading
import datetime
import trace
import random
from tkinter import *
from time import sleep
import json


user = []
bot = []
response = ''
customerId = 0
StartTime = 0
lastReplyTime = 0

intents = json.loads(open('intents.json').read())

def chatbot_response1():
    query = layout1.EntryBox.get().strip()
    layout1.EntryBox.delete(0, END)
    global user
    global bot
    user.append(query)  # Recording user response

    if query != '':
        # Session Recording
        # Resetting lastReplyTime
        c = datetime.datetime.now()
        current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
        global lastReplyTime
        lastReplyTime = current_time
        print("chat_response: ", lastReplyTime)

    def enterquery():
        layout1.ChatLog.config(state=NORMAL)
        layout1.ChatLog.insert(END, "You: " + query + '\n\n')
        layout1.ChatLog.config(state=DISABLED)

    def enterresponse():
        global response
        userID = '123'
        ints = chatbot1.predict_class(query, chatbot1.model)
        response = chatbot1.getResponse(ints, userID, intents)
        layout1.ChatLog.config(state=NORMAL)
        layout1.ChatLog.insert(END, "AssistBot: " + str(response) + '\n\n')
        layout1.ChatLog.config(state=DISABLED)
        layout1.EntryBox.delete(0, END)
        layout1.ChatLog.yview(END)
        voice1.speak(response)

    global response
    enterquery()
    enterresponse()
    bot.append(response)  # Recording bot response


def push_to_mongodb(uniqueSessionId):
    global StartTime
    global customerId
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = "Metadata"
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    dataBase = client[DB_NAME]
    COLLECTION_NAME = "Session_Info"
    collection = dataBase[COLLECTION_NAME]
    record = {'SessionId': uniqueSessionId,
              'StartTime': StartTime,
              'EndTime': datetime.datetime.now(),
              'CustomerId': customerId,
              'CustomerStatus': 'Active',
              'DataFileName': 'abc.txt'}
    collection.insert_one(record)
    print(record)
    COLLECTION_NAME = "Session_Data"
    collection = dataBase[COLLECTION_NAME]
    record2 = {'SessionId': uniqueSessionId,
               'CustomerId': customerId,
               'UserConv': user,
               'BotResp': bot,
               }
    collection.insert_one(record2)
    print(record2)


def ideal(uniqueSessionId):
    global lastReplyTime
    global customerId
    global user
    global bot
    global StartTime
    customerId = random.random()
    c = datetime.datetime.now()
    StartTime = c
    print("uniqueSessionId : ", uniqueSessionId)
    while True:
        c = datetime.datetime.now()
        current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
        print("While lastReplyTime: ", int(lastReplyTime), " ", current_time)
        sleep(1)
        if (lastReplyTime + 20) == current_time:
            print("ideal")
            endmsg = '''Since there is no response from your end, I will have to end this conversation now.
                      I appreciate your time and patients. Please text or speak to me if you need my help. Goodbye'''
            layout1.ChatLog.config(state=NORMAL)
            layout1.ChatLog.insert(END, "AssistBot: " + str(endmsg) + '\n\n')
            layout1.ChatLog.config(state=DISABLED)
            voice1.speak(endmsg)
        if (lastReplyTime + 40) == current_time:
            push_to_mongodb(uniqueSessionId)
            print("Closed")
            layout1.main.destroy()
            global stop
            stop = False
            break


def session():
    c = datetime.datetime.now()
    current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second + c.microsecond
    uniqueSessionId = current_time
    t1 = threading.Thread(target=ideal, args=(uniqueSessionId,))
    t1.start()


session()  # Initializing session