import datetime
from time import sleep
from threading import *

bot_reply_time = 0
x = 1


class chat(Thread):
    def run(self):
        y = 0
        global bot_reply_time
        while x > 0:
            a = input("Q>")
            print("Bot replay: Hi")
            c = datetime.datetime.now()
            bot_reply_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
            if y == 0:
                y += 1
                i = ideal()
                i.start()


class ideal(Thread):
    def run(self):
        global x
        global bot_reply_time
        print("hi")
        while True:
            last_bot_reply_time = bot_reply_time
            c = datetime.datetime.now()
            current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
            if (bot_reply_time + 10 == current_time and bot_reply_time == last_bot_reply_time):
                x = 0
                print("session expired")
                print(bot_reply_time)
                print(current_time)
                break


c = chat()
c.start()
