import pandas as pd
from pymongo import MongoClient
import csv
import json
df=pd.read_csv("Personal_loan_dataset.csv")
del df["Unnamed: 0"]
client = MongoClient('localhost', 27017) #connection string
db = client['ChatBotDB']       #databse name
collection = db['CustomerInfo'] #collection name
def insertToMongo(df):
    collection.insert_many(df.to_dict("records"))
filter={"Age":29,"Gender":"Male"} #gives output for age 29 and gender male only
def query(filter):
    result= collection.find(filter)

    result_df= pd.DataFrame(result)

    print(result_df.head())
insertToMongo(df)
query(filter)
