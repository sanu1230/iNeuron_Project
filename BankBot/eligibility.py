import pandas as pd
from datetime import date, datetime


def eligibility_input(user):
    eli_inputs = []
    for x in user:
        x.lower()
        if "my name is" in x:
            name = x.replace("my name is ", '')
            eli_inputs.append(name)
        elif "date is" in x:
            dob = [int(s) for s in x.split() if s.isdigit()]
            eli_inputs.append(dob)
        elif "net income" in x:
            netincome = [int(s) for s in x.split() if s.isdigit()]
            eli_inputs.append(netincome)
        elif "total expenses" in x:
            totalexp = [int(s) for s in x.split() if s.isdigit()]
            eli_inputs.append(totalexp)
    print("Inputs", eli_inputs)
    return eli_inputs


def calculateAge(birthDate):
    today = date.today()
    age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
    print("Age: ", age)
    return age

def age_eli_prob(age):
    if int(age) < 20:
        age_prob = 0
    elif 20 < int(age) < 25:
        age_prob = 30
    elif 25 < int(age) < 45:
        age_prob = 70
    elif 45 < int(age) < 55:
        age_prob = 50
    elif 55 < int(age) < 65:
        age_prob = 35
    print("Age Pb: ", age_prob)
    return age_prob


def eligibility_check(user):
    global savings
    inputs = eligibility_input(user)
    dob = inputs[1]
    str1=""
    for x in dob:
        str1+=str(x)
    dobs = str1
    c_date = datetime.strptime(dobs, '%d%m%Y').strftime('%Y-%m-%d')
    db = pd.to_datetime(c_date)  # convert into datetime
    age = calculateAge(db)
    age_prob = age_eli_prob(age)
    income = inputs[2][0]
    expenses = inputs[3][0]
    savings = int(income) - int(expenses)
    savings_prob = (int(savings)*100)/int(income)
    avg_score = (age_prob + savings_prob)/2
    print("Avg Elg: ", avg_score)
    return avg_score


def eligibility_response(user):
    global savings
    avg_score = eligibility_check(user)
    if avg_score < 50:
        response = "I regret to inform you that, currently you are not eligible for the loan. " \
                   "However, you may try again after few months. Is there anything else I may help you with"
    else:
        amount = (int(savings)*12)*3
        formatted_currency = "{:,.2f}".format(amount)
        response = "Congratulations!!, you are eligible for a loan, up to '\u20B9'{}. " \
                   "Would you like to apply for the loan".format(formatted_currency)
    print(response)
    return response