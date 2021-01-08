from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
from chatterbot.logic import LogicAdapter
import os

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

app = Flask(__name__)
bot = ChatBot(
    "AssistBot",
    logic_adapters=[
         {
            "import_path": "chatterbot.logic.BestMatch",
            "import_path": "chatterbot.logic.TimeLogicAdapter",
            "import_path": "chatterbot.logic.MathematicalEvaluation",
            "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
            "response_selection_method": "chatterbot.response_selection.get_first_response",
        },
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'threshold': 0.65,
            'default_response': ['I am sorry, but I do not understand.',
                                "Sorry I dont understand you, try rephrasing your sentence"]}],
    preprocessors = ['chatterbot.preprocessors.clean_whitespace', 'chatterbot.preprocessors.unescape_html'], 
    filters= ['chatterbot.filters.RepetitiveResponseFilter'],
    storage_adapter="chatterbot.storage.SQLStorageAdapter")
    
ltrainer = ListTrainer(bot)

for knowledeg in os.listdir('base'):
	BotMemory = open('base/'+ knowledeg, 'r').readlines()
	ltrainer.train(BotMemory)

trainer = ChatterBotCorpusTrainer(bot)
trainer.train("chatterbot.corpus.english")
trainer.train("data/faq.yml")
trainer.train("data/greet.yml")

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    botresponse = str(bot.get_response(userText))
    return botresponse


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001, threaded=True)
