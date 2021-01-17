








# def exitpg():  # to exit the program
#     global stop_repeatl
#     try:
#         pushconv_to_mongodb(uniqueSessionId)
#         print("Closed")
#         stop_repeatl = True
#         sys.exit()
#         # main.destroy()
#     except Exception as e:
#         print(e)
#         print("Closed")
#         stop_repeatl = True
#         sys.exit()
#         # main.destroy()


# def chatbot_response():
#     global lastReplyTime, user
#     query = request.form["user_input"]
#     print("User Input: ", query)
#
#     def replytime():
#         global lastReplyTime, user, bot
#         # Session Recording
#         # Resetting lastReplyTime
#         c = datetime.now()
#         current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
#         lastReplyTime = current_time
#         print("chat_response: ", lastReplyTime)
#         print("user list: ", user)
#         print("bot list: ", bot)
#         return lastReplyTime
#
#     lastReplyTime = replytime()
#     user = user_list()
#     return user


# def exitpg():  # to exit the program
#     global stop_repeatl
#     try:
#         pushconv_to_mongodb(uniqueSessionId)
#         stop_repeatl = True
#         print("Sp off")
#     except Exception as e:
#         print(e)
#         stop_repeatl = True
#         print("Sp off")



# def ideal(uniqueSessionId):
#     global lastReplyTime, customerId, user, bot, StartTime, stop_repeatl
#     customerId = random.random()
#     c = datetime.now()
#     StartTime = c
#     print("uniqueSessionId : ", uniqueSessionId)
#     while True:
#         c = datetime.now()
#         current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second
#         print("While lastReplyTime: ", int(lastReplyTime), " ", current_time)
#         sleep(1)
#         if (lastReplyTime + 30) == current_time:
#             print("ideal")
#             endmsg = '''Since there is no response from your end, I will have to end this conversation now.
#                       I appreciate your time and patients. Please text or speak to me if you need my help. Goodbye'''
#             speak(endmsg)
#         if (lastReplyTime + 50) == current_time:
#             exitpg()
#             sleep(5)
#             print("sys exit")
#             speak("GoodBye")
#             print("GoodBye")
#             os.kill(os.getpid(), signal.SIGINT)
#             # sys.exit()


# def session():
#     global uniqueSessionId
#     c = datetime.now()
#     current_time = (c.hour * 60 * 60) + (c.minute * 60) + c.second + c.microsecond
#     uniqueSessionId = current_time
#     t1 = threading.Thread(target=ideal, args=(uniqueSessionId,))
#     t1.daemon = True
#     t1.start()
#
#
# session()  # Initializing session
#
# URL = "http://127.0.0.1:5000/process"



# def speak(word):
#     global sr
#     # global engine
#     engine = pp.init()
#     # Setting up voice rate
#     engine.setProperty('rate', 190)
#     voices = engine.getProperty('voices')  # get all voices
#     engine.setProperty('voice', voices[1].id)
#     store = sr.energy_threshold
#     sr.energy_threshold = 9001  # note I'm not sure what the maximum value is
#     engine.say(word)
#     engine.runAndWait()
#     engine.stop()
#     sr.energy_threshold = store



# def speech_query():
#     global sr, URl
#     sr = sp.Recognizer()
#     sr.pause_threshold = 1
#     with sp.Microphone() as m:
#         sr.adjust_for_ambient_noise(m, duration=0.2)
#         try:
#             audio = sr.listen(m)
#             query = sr.recognize_google(audio, language="eng-in")
#             print(query)
#             values = {"user_query": query}
#             val = json.dumps(values)
#             requests.post(URL, json=val).json()
#             # driver = webdriver.Remote('http://127.0.0.1:5000/process')
#             # # driver.maximize_window()
#             # # driver.get("http://127.0.0.1:5000/process")
#             # input_box = driver.find_element_by_xpath('//*[@id="v_input"]')
#             # input_box.send_keys(str(query))
#             # send_button = driver.find_element_by_xpath('/html/body/div[1]/form/div/button')
#             # send_button.click()
#         except Exception as e:
#             print(e)
#             print("not recognize")



# global stop_repeatl
# stop_repeatl = False


# def repeatl():
#     global stop_repeatl
#     while stop_repeatl == False:
#         speech_query()
#         print('speech thread running')
#     else:
#         print("threads killed")
#
#
# r = threading.Thread(target=repeatl)
# r.daemon = True
# r.start()

# def intro():
#     print('intro thread running')
#     intromsg = "Hi, I am AssistBot. Your customer service agent. How may I help you?"
#     speak(intromsg)
#     bot.append(intromsg)
#
