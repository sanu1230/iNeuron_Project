from tkinter import *
#import voice1

import threading
import chatbot1

# global main
main = Tk() #blank screen
main.geometry("350x600")    #screen dimensions
main.resizable(width=True, height=True) #
main.title("AssistBot") #header


img = PhotoImage(file="images.png") #img file to add
photoL = Label(main, image=img) #giving label
photoL.pack(pady=5) #dimensions of labe

frame = Frame(main)
frame.pack()
# Creating a message box in frame
# global ChatLog
ChatLog = Text(frame, bd=0, bg="white", height="15", width="35", font="Bahnschrift", wrap=WORD)
ChatLog.pack(side=LEFT, fill=BOTH, pady=10)
ChatLog.config(state=DISABLED)

# Creating scrollbar in frame
sc = Scrollbar(frame, command=ChatLog.yview)
ChatLog['yscrollcommand'] = sc.set
sc.pack(side=RIGHT, fill=Y)

# Creating text field
# global EntryBox
EntryBox = Entry(main, width="23", font=("Bahnschrift", 20))
EntryBox.pack(pady=10)

import voice1
from voice1 import speak
from voice1 import speech_query
#from chatbot1 import chatbot_response
import chatbot2
#import chatbot.chatbot_response
# Creating button
btn = Button(main, text="Send", width="30", font=("Bahnschrift", 15), command=chatbot2.chatbot_response1)
btn.pack()


def enter_function(event):
    btn.invoke()


# going to bind main window with enter key
main.bind('<Return>', enter_function)


import voice1
from voice1 import speak
from voice1 import speech_query


def intro():
    intromsg = "Hi, I am AssistBot. Your customer service agent. How may I help you?"
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "AssistBot: " + intromsg + '\n\n')
    ChatLog.config(state=DISABLED)
    voice1.speak(intromsg)


def repeatl():
    global stop
    while stop:
        voice1.speech_query()

stop = True
r = threading.Thread(target=repeatl)
r.start()

n = threading.Thread(target=intro)
n.start()

main.mainloop()