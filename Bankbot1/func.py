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
words = pickle.load(open('api/model/words.pkl', 'rb'))
classes = pickle.load(open('api/model/classes.pkl', 'rb'))


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


def predict_class(sentence, words, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    # print('res: ', res)
    ERROR_THRESHOLD = 0.4
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    print('predict_class: ', results)
    return_list = []
    if not results:
        return_list.append({"intent": "defaultfallback"})
    else:
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    print('predict_class1: ', return_list)
    return return_list


yes = ["yes", "ok", "yep", "okay", "fine", "cool", "sure", "all right", "yes please", "go ahead", "positive", "ok man",
       "of course", "correct"]
no = ["no", "nope", "never", "na", "naa", "I do not", "you cannot", "No please", "stop", "no man", "wrong"]


context = {}
all_conv_tags = []
recheck_tag = []
recheck_tag_yes = ["personal_loan_apply_yes", "ploan_apply_contact_yes"]
recheck_tag_no = ["personal_loan_apply_no", "ploan_apply_contact_no"]

def reconfirm(query):
    response = "You have mentioned --{}--. Is this information correct, please confirm in -Yes- or -No-".format(query)
    return response


def getTag(query, ints, cTag):
    global all_conv_tags
    if query in yes:
        con = context[cTag]
        tag = con[0] + "_yes"
    elif query in no:
        con = context[cTag]
        tag = con[0] + "_no"
    else:
        tag = ints[0]['intent']
    print('getTag1: ', tag)
# ----------------------------
    if tag in recheck_tag_yes:
        tag = recheck_tag[-1]
    elif tag in recheck_tag_no:
        tag = all_conv_tags[-2]
    else:
        tag = tag

    print('getTag2: ', tag)
    all_conv_tags.append(tag)
    print("all_conv_tags - ", all_conv_tags)
    return tag


def tagPath_checker(tag, all_conv_tags, story_conv):
    for conv in story_conv:
        if all_conv_tags[-2] in conv:
            nextTag = conv[all_conv_tags[-2]]
            if tag == nextTag or tag == all_conv_tags[-2]:
                return tag
            else:
                tag = all_conv_tags[-2]
                return tag


def contextResponse(query, tag, intents_json, cTag):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if "recheck" in i:
                if query == 'no' and (i['context_filter'] == context[cTag]):
                    response = random.choice(i['responses'])
                    recheck_tag.append(tag)
                    print("recheck_tag", recheck_tag)
                    return response
                elif query not in yes and (i['context_filter'] == context[cTag]):
                    response = reconfirm(query)
                    recheck_tag.append(tag)
                    print("recheck_tag", recheck_tag)
                    return response
                elif 'context_set' in i:
                    context[cTag] = i['context_set']
                    response = random.choice(i['responses'])
                    print(context)
                    return response
                elif 'context_filter' not in i or \
                        (cTag in context and 'context_filter' in i and i['context_filter'] == context[cTag]):
                    response = random.choice(i['responses'])
                    print(context)
                    return response

            elif 'context_set' in i:
                context[cTag] = i['context_set']
                response = random.choice(i['responses'])
                print(context)
                return response
            elif 'context_filter' not in i or \
                    (cTag in context and 'context_filter' in i and i['context_filter'] == context[cTag]):
                response = random.choice(i['responses'])
                print(context)
                return response



