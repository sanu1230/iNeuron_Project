import pymongo
import re
import pandas as pd
from pymongo import MongoClient
from datetime import datetime



def status_input(user):
    stat_input = []
    for i in user:
        i = i.lower()
        if "application number" in i:
            app_id = re.findall('[0-9]+', i)
            app_id = 'pl' + str(app_id[0])
            stat_input.append(app_id)

    stat_dict = {"ApplicationNo": stat_input[-1]}
    return stat_dict

# pl58995

def pulldata(app_num):
    global res
    db_client = MongoClient()
    db = db_client.Personal_Loan
    db = db.Personal_Loan_Applications
    res = db.find_one(app_num)
    print(res)
    return res

def application_status(user):
    input = status_input(user)
    app_data = pulldata(input)
    app_date = app_data['Application_Date']
    app_date = pd.to_datetime(app_date)
    date_today = datetime.now().date()
    days_diff = date_today - app_date
    result = days_diff.days
    if result <= 15:
        response = "Your application is in process. Please wait at least 10 working days. " \
                   "Just before you leave I would like to inform you that our bank offers you a 'Accidental Policy' " \
                   "worth 5 lacs. Would you like to know more about the offer"
        return response
    elif result > 15:
        response = "I regret to inform you that your application has been denied. " \
                   "Please wait at least 60 working days before you apply again. " \
                   "Just before you leave I would like to inform you that our bank offers you a 'Accidental Policy' " \
                   "worth 5 lacs. Would you like to know more about the offer"
        return response


