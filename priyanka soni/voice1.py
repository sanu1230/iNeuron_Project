import pyttsx3 as pp
import speech_recognition as sp
#import main1
#from __main1__ import *
#import globals
import layout1
from tkinter import *
import chatbot2

engine = pp.init()

voices = engine.getProperty('voices')  # get all voices
engine.setProperty('voice', voices[1].id)


def speak(word):
    engine.say(word)
    engine.runAndWait()


from layout1 import EntryBox
#from chatbot1 import chatbot_response
import chatbot1

def speech_query():
    sr = sp.Recognizer()
    sr.pause_threshold = 1
    with sp.Microphone() as m:
        try:
            audio = sr.listen(m)
            query = sr.recognize_google(audio, language="eng-in")
            layout1.EntryBox.delete(0,END)
            layout1.EntryBox.insert(0, query)
            chatbot2.chatbot_response1()
        except Exception as e:
            print(e)
            print("not recognize")
