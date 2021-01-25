import yaml
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

with open(r'data/faq.yaml') as file:
    faq_list = yaml.load(file, Loader=yaml.FullLoader)

lemmatizer = WordNetLemmatizer()


def faq_clean_up(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence_words = tokenizer.tokenize(str(sentence))
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def faq_response(query):
    list1 = faq_clean_up(query)
    scr = 0
    quest = ''
    response = ''

    for item in faq_list['conversations']:
        tk_item = faq_clean_up(item[0])
        list2 = tk_item
        # Percentage similarity of lists
        # using "|" operator + "&" operator + set()
        res = len(set(list1) & set(list2)) / float(len(set(list1) | set(list2))) * 100
        if res > scr:
            scr = res
            quest = item[0]
            response = item[1]
            match_perc1 = SequenceMatcher(None, query, quest).ratio()
            # print(match_perc)

    if match_perc1 > 0.60:
        return response

# query = "if there is issue during withdrawal or part prepayment"
# res = faq_response(query)
# print(res)

