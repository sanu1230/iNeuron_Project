import nltk
import pandas as pd
from datetime import date, datetime
from nltk.corpus import stopwords
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import datetime
from time import sleep
import pyttsx3 as pp
import speech_recognition as sp
import threading
import pymongo
from tkinter import *
from func import *
import sys
import time
from eligibility import *

lemmatizer = WordNetLemmatizer()


words = []
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        text_token = nltk.word_tokenize(pattern)
        token_without_sw = [word for word in text_token if word not in stopwords.words()]
        words.extend(token_without_sw)
        # adding documents
        documents.append((token_without_sw, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# print(len(documents), "documents")
# print(len(classes), "classes", classes)
# print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# initializing training data, Creating X and y data (bag and output row)
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []  # initializing bag of words
    pattern_words = doc[0]  # list of tokenized words for the pattern
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        # create our bag of words array with 1, if word match found in current pattern
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])


# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")


# Create model - 3 layers.
# First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")


user = []
bot = []
response = ''
customerId = 0
StartTime = 0
global lastReplyTime
c = datetime.now()
lastReplyTime = (c.hour * 60 * 60) + (c.minute * 60) + c.second  # lastreplytime is set on current time
print("No_Conversation: ", lastReplyTime)


def enterquery_in_entrybox(query):
    global EntryBox
    EntryBox.delete(0, END)
    EntryBox.insert(0, query)


def getquery():
    query = EntryBox.get().strip()
    EntryBox.delete(0, END)
    return query


def enterquery(query):
    global ChatLog, user
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "You: " + query + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)
    EntryBox.delete(0, END)
    user.append(query)  # Recording user response



def enterresponse(msg):
    global ChatLog
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "AssistBot: " + msg + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)


def exitpg():  # to exit the program
    global stop_repeatl
    push_to_mongodb(uniqueSessionId)
    print("Closed")
    stop_repeatl = True
    main.destroy()


def enter_proper_response(query):
    global response, bot, user
    userID = '123'
    ints = predict_class(query, model)
    if ints[0]['intent'] == "goodbye":  # or "anything_else_no"
        response = getResponse(query, ints, userID, intents)
        enterresponse(response)
        speak(response)
        bot.append(response)  # Recording bot response
        exitpg()
    else:
        response = getResponse(query, ints, userID, intents)
        if response == "calc_eli":
            response = eligibility_response(user)
        else:
            response = response

        enterresponse(response)
        speak(response)
        bot.append(response)  # Recording bot response


def chatbot_response():
    query = getquery()
    enterquery(query)
    global lastReplyTime

    def replytime():
        global lastReplyTime
        # Session Recording
        # Resetting lastReplyTime
        c = datetime.now()
        current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
        lastReplyTime = current_time
        print("chat_response: ", lastReplyTime)
        return lastReplyTime

    if query != '':
        lastReplyTime = replytime()
        enter_proper_response(query)
    else:
        lastReplyTime = replytime()
        emptymsg = "Please text or say your query. I'll be glad to help you."
        enterresponse(emptymsg)
        speak(emptymsg)


def push_to_mongodb(uniqueSessionId):
    global StartTime, customerId, user, bot
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
    DB_NAME = "Metadata"
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)
    dataBase = client[DB_NAME]
    COLLECTION_NAME = "Session_Info"
    collection = dataBase[COLLECTION_NAME]
    record = {'SessionId': uniqueSessionId,
              'StartTime': StartTime,
              'EndTime': datetime.now(),
              'CustomerId': customerId,
              'CustomerStatus': 'Active',
              'DataFileName': 'abc.txt'}
    collection.insert_one(record)
    print("Record 1", record)
    COLLECTION_NAME = "Session_Data"
    collection = dataBase[COLLECTION_NAME]
    record2 = {'SessionId': uniqueSessionId,
               'CustomerId': customerId,
               'UserConv': user,
               'BotResp': bot,
               }
    collection.insert_one(record2)
    print("Record 2", record2)


def ideal(uniqueSessionId):
    global lastReplyTime, customerId, user, bot, StartTime, stop_repeatl
    customerId = random.random()
    c = datetime.now()
    StartTime = c
    print("uniqueSessionId : ", uniqueSessionId)
    while True:
        c = datetime.now()
        current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
        print("While lastReplyTime: ", int(lastReplyTime), " ", current_time)
        sleep(1)
        if (lastReplyTime + 30) == current_time:
            print("ideal")
            endmsg = '''Since there is no response from your end, I will have to end this conversation now.
                      I appreciate your time and patients. Please text or speak to me if you need my help. Goodbye'''
            enterresponse(endmsg)
            speak(endmsg)
        if (lastReplyTime + 50) == current_time:
            exitpg()


def session():
    global uniqueSessionId
    c = datetime.now()
    current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second + c.microsecond
    uniqueSessionId = current_time
    t1 = threading.Thread(target=ideal, args=(uniqueSessionId,))
    t1.daemon = True
    t1.start()


session()  # Initializing session

# ---------------------------------- For Voice -------------------------------- ##
engine = pp.init()

voices = engine.getProperty('voices')  # get all voices
engine.setProperty('voice', voices[1].id)


def speak(word):
    global sr
    store = sr.energy_threshold
    sr.energy_threshold = 9001  # note I'm not sure what the maximum value is
    engine.say(word)
    engine.runAndWait()
    sr.energy_threshold = store


def speech_query():
    global sr
    sr = sp.Recognizer()
    sr.pause_threshold = 1
    with sp.Microphone() as m:
        sr.adjust_for_ambient_noise(m, duration=0.2)
        try:
            audio = sr.listen(m)
            query = sr.recognize_google(audio, language="eng-in")
            enterquery_in_entrybox(query)
            # enterquery()
            chatbot_response()
        except Exception as e:
            print(e)
            print("not recognize")


# -----------------------------GUI------------------------------------- #

main = Tk()
main.geometry("350x600")
main.resizable(width=True, height=True)
main.title("AssistBot")


img = PhotoImage(file="images.png")
photoL = Label(main, image=img)
photoL.pack(pady=5)

frame = Frame(main)
frame.pack()

# Creating a message box in frame
ChatLog = Text(frame, bd=0, bg="white", height="13", width="35", font="Bahnschrift", wrap=WORD)
ChatLog.pack(side=LEFT, fill=BOTH, pady=10)
ChatLog.config(state=DISABLED)

# Creating scrollbar in frame
sc = Scrollbar(frame, command=ChatLog.yview)
ChatLog['yscrollcommand'] = sc.set
sc.pack(side=RIGHT, fill=Y)

# Creating text field
EntryBox = Entry(main, width="23", font=("Bahnschrift", 20))
EntryBox.pack(pady=10)

# Creating button
btn = Button(main, text="Send", width="30", font=("Bahnschrift", 15), command=chatbot_response)
btn.pack()


def enter_function(event):
    btn.invoke()


# going to bind main window with enter key
main.bind('<Return>', enter_function)


def intro():
    print('intro thread running')
    intromsg = "Hi, I am AssistBot. Your customer service agent. How may I help you?"
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "AssistBot: " + intromsg + '\n\n')
    ChatLog.config(state=DISABLED)
    speak(intromsg)


global stop_repeatl
stop_repeatl = False


def repeatl():
    global stop_repeatl
    while stop_repeatl == False:
        speech_query()
        print('speech thread running')
    else:
        print("threads killed")


r = threading.Thread(target = repeatl)
r.daemon = True
r.start()


n = threading.Thread(target=intro)
n.daemon = True
n.start()

main.mainloop()

