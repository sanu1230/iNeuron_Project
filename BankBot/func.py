import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from keras.models import load_model

model = load_model('chatbot_model.h5')

intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


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
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


yes = ["yes", "ok", "yep", "okay", "fine", "cool", "sure", "all right", "yes please", "go ahead", "positive", "ok man",
       "of course"]
no = ["no", "nope", "never", "na", "naa", "I do not", "you cannot", "No please", "stop", "no man"]


context = {}
all_conv_tags = []
recheck_tag = []
recheck_tag_yes = ["personal_loan_apply_yes"]
recheck_tag_no = ["personal_loan_apply_no"]

def reconfirm(query):
    response = "You have mentioned '{}'. Is this information correct, please confirm in 'Yes' or 'No'".format(query)
    return response


def getTag(query, ints, userID):
    global all_conv_tags
    if query in yes:
        con = context[userID]
        tag = con[0] + "_yes"
    elif query in no:
        con = context[userID]
        tag = con[0] + "_no"
    elif ints == []:
        tag = 'defaultfallback'
    else:
        tag = ints[0]['intent']

    if tag in recheck_tag_yes:
        tag = recheck_tag[-1]
    elif tag in recheck_tag_no:
        tag = all_conv_tags[-2]

    print('Tag -', tag)
    all_conv_tags.append(tag)
    print("all_conv_tags - ", all_conv_tags)
    return tag




def contextResponse(query, tag, userID, intents_json, user):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if "recheck" in i:
                if query == 'no':
                    response = random.choice(i['responses'])
                    recheck_tag.append(tag)
                    print("recheck_tag", recheck_tag)
                    return response
                elif query != 'yes':
                    response = reconfirm(query)
                    recheck_tag.append(tag)
                    print("recheck_tag", recheck_tag)
                    return response
                elif 'context_set' in i:
                    context[userID] = i['context_set']
                    response = random.choice(i['responses'])
                    print(context)
                    return response
                elif 'context_filter' not in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                    response = random.choice(i['responses'])
                    print(context)
                    return response

            elif 'context_set' in i:
                context[userID] = i['context_set']
                response = random.choice(i['responses'])
                print(context)
                return response
            elif 'context_filter' not in i or \
                    (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                response = random.choice(i['responses'])
                print(context)
                return response

