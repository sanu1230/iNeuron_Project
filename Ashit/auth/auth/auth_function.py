import re
import pymongo
import pandas as pd
from pprint import pprint
from pymongo import MongoClient
from eligibility import *


def authcheck(data):
    aut_input = []
    aut_id = []
    aut_dob = []
    name_input = []

    for i in data:
        i = i.lower()
        if "customer_id" in i:
            customer_id = i.replace("my UserID is ", '')
            aut_id.append(customer_id)
        elif "my first name is" in i:
            nam = i.replace("my first name is ", '')
            name_input.append([nam])
        elif "date of birth is" in i:
            dob = re.findall('[0-9]+', i)
            aut_dob.append(dob)

    aut_input.append(aut_id)
    aut_input.append(name_input)
    aut_input.append(aut_dob)
    print("inpt -", aut_input)
    return aut_input


def processing(data):
    inpt = authcheck(data)
    customer_id = inpt[0]     # [ we should feed the input to Chatbot sequence wise like customer_id, name, and dob ]
    name = inpt[1]
    dob = inpt[2]
    in_dict = {"Customer_id": customer_id, "First_name": name, "Date_of_birth": dob}
    print("input_dict:", in_dict)
    in_df = pd.DataFrame(in_dict, index=[0])
    print('input_df: ', in_df)
    return in_df


def main(aut_input):
    global res
    db_client = MongoClient()
    db = db_client.AIBankbot
    db = db.Personal_Loan_Customers
    res = db.find_one({"Customer_id": aut_input[0], "First_name": aut_input[1], "Date_of_birth": aut_input[2]})
    pprint(res)


def result(res):
    if res == None:
        return False
    else:
        return True,
