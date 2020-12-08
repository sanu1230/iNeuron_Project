import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import datetime
from time import sleep
import pyttsx3 as pp
import speech_recognition as sp
import threading
import pymongo
from tkinter import *
import os
import sys
# import eligibility_class




words = []
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# initializing training data, Creating X and y data (bag and output row)
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = [] # initializing bag of words
    pattern_words = doc[0]  # list of tokenized words for the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words] # lemmatize each word - create base word, in attempt to represent related words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)   # create our bag of words array with 1, if word match found in current pattern

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])


# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
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

from keras.models import load_model
model = load_model('chatbot_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Once again, we need to extract the information from our files.

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


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
    ERROR_THRESHOLD = 0.9
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


context = {}
data_dict = {}

def getResponse(sentence,ints, userID, intents_json):
    if ints == []:
        tag = 'defaultfallback'
    else:
        tag = ints[0]['intent']

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            if 'context_set' in i:
                context[userID] = i['context_set']
            
            if not 'context_filter' in i or \
                    (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                result = random.choice(i['responses'])
            
            #ADITYA 6/12
            if 'data' in i:
                match = re.search(i['data'],sentence).group()
                print('Adding ',match,' to',tag )
                data_dict[tag] = match
    
    print(data_dict)
    return result

def pushmsg(msg):
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "AssistBot: " + msg + '\n\n')
    ChatLog.config(state=DISABLED)
    speak(msg)

def entermsg():
    EntryBox.delete(0, END)
    if EntryBox.get() != '':
        userMessage = EntryBox.get()
    return userMessage

user = []
bot = []
def init():
    global globallist
    globallist = [user, bot]

def chatbot_response():
    query = EntryBox.get().strip()
    EntryBox.delete(0, END)
    # init()
    if query != '':
        # Session Recording
        # Resetting lastReplyTime
        global lastReplyTime
        c = datetime.datetime.now()
        current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
        lastReplyTime = current_time

    def enterquery():
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + query + '\n\n')
        ChatLog.config(state=DISABLED)
        # Recording user response
        user.append(str(query))

    def enterresponse():
        userID = '123'
        ints = predict_class(query, model)
        response = getResponse(query,ints, userID, intents)
        bot.append(str(response))
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "AssistBot: " + response + '\n\n')
        ChatLog.config(state=DISABLED)
        EntryBox.delete(0, END)
        ChatLog.yview(END)
        speak(response)

    enterquery()
    enterresponse()


def ideal(uniqueSessionId):
    global lastReplyTime
    global customerId
    init()
    customerId = random.random()
    c = datetime.datetime.now()
    StartTime = c
    lastReplyTime = (c.hour * 60 * 60) + (c.minute * 60) + c.second
    print("uniqueSessionId : ",uniqueSessionId)
    while True:
        c = datetime.datetime.now()
        current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
        print("While lastReplyTime:",lastReplyTime,current_time)
        sleep(1)
        if (lastReplyTime + 10 <= current_time):
            print("ideal")
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
            COLLECTION_NAME = "Session_Data"
            collection = dataBase[COLLECTION_NAME]
            record2 = {'SessionId': uniqueSessionId,
                      'CustomerId': customerId,
                      'UserConv': user,
                      'BotResp': bot,
                    }
            collection.insert_one(record2)
            break


def session():
    c = datetime.datetime.now()
    current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second + c.microsecond
    uniqueSessionId = current_time
    t1 = threading.Thread(target=ideal,args=(uniqueSessionId,))
    t1.start()

# Initializing session
session()


# ---------------------------------- For Voice -------------------------------- ##
# engine = pp.init()

# voices = engine.getProperty('voices')  # get all voices
# engine.setProperty('voice', voices[1].id)


# def speak(word):
#     engine.say(word)
#     engine.runAndWait()


# def speech_query():
#     sr = sp.Recognizer()
#     sr.pause_threshold=1
#     with sp.Microphone() as m:
#         try:
#             audio = sr.listen(m)
#             query = sr.recognize_google(audio, language="eng-in")
#             EntryBox.delete(0, END)
#             EntryBox.insert(0, query)
#             chatbot_response()
#         except Exception as e:
#             print(e)
#             print("not recognize")


#-----------------------------GUI-------------------------------------#

main = Tk()
main.geometry("350x650")
main.resizable(width=True, height=True)
main.title("AssistBot")


# img = PhotoImage(file="images.png")
# photoL = Label(main, image=img)
# photoL.pack(pady=5)

frame = Frame(main)
frame.pack()

# Creating a message box in frame
ChatLog = Text(frame, bd=0, bg="white", height="15", width="35", font="Bahnschrift", wrap=WORD)
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


def repeatl():
    while True:
        speech_query()


def intro():
    intro = "Hi, I am AssistBot. Your customer service agent. How may I help you?"
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "AssistBot: " + intro + '\n\n')
    ChatLog.config(state=DISABLED)
    # speak(intro)


# r = threading.Thread(target=repeatl)
# r.start()
# n = threading.Thread(target=intro)
# n.start()

main.mainloop()
