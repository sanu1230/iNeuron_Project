import pandas as pd
from datetime import date, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
from time import sleep

df = pd.read_csv('data/Eligibility_data.csv')

# Creating model function

# one-hot-encoding
one_hot_features = ['Gender', 'Eligibility']

# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(df[one_hot_features], drop_first=True)
# Convert Categorical to Numerical for default column
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

# Replacing categorical columns with dummies
fdf = df.drop(one_hot_features, axis=1)
fdf = pd.concat([fdf, one_hot_encoded], axis=1)

# Label encoding the Company Rank
fdf['Company'] = fdf['Company'].map({'Rank 1': 0, 'Rank 2': 1, 'Rank 3': 2})

# splitting X  &  y
X = fdf.drop('Eligibility_yes', 1)
y = fdf['Eligibility_yes']

# splitting in train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)

# Creating model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# test model
y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model
file_name = 'eligibility_model.sav'
pickle.dump(rf_model, open(file_name, 'wb'))

# load the model from disk
model_eli = pickle.load(open(file_name, 'rb'))



def find_eligibility(input):
    pred = model_eli.predict(input)
    return pred


def eligibility_input(user):
    eli_inputs = []
    name_inputs = []
    gen_inputs = []
    dob_inputs =[]
    income_inputs =[]
    expense_inputs = []
    company_inputs = []
    experience_inputs = []

    for x in user:
        x.lower()
        if "my name is" in x:
            name = x.replace("my name is ", '')
            name_inputs.append([name])
            gen = gender_code(name)
            gen_inputs.append([gen])
        elif "date of birth is" in x or "date is" in x:
            dob = [int(s) for s in x.split() if s.isdigit()]
            dob_inputs.append(dob)
        elif "net income" in x:
            netincome = [int(s) for s in x.split() if s.isdigit()]
            income_inputs.append(netincome)
        elif "total expenses" in x:
            totalexp = [int(s) for s in x.split() if s.isdigit()]
            expense_inputs.append(totalexp)
        elif "company name" in x or "business name" in x:
            if "company name" in x:
                company_name = x.replace('company name ', '')
                company_inputs.append([company_name])
            else:
                company_name = x.replace('business name ', '')
                company_inputs.append([company_name])
        elif "total experience" in x:
            exp = [int(s) for s in x.split() if s.isdigit()]
            experience_inputs.append(exp)

    eli_inputs.append(name_inputs[-1])
    eli_inputs.append(gen_inputs[-1])
    eli_inputs.append(dob_inputs[-1])
    eli_inputs.append(income_inputs[-1])
    eli_inputs.append(expense_inputs[-1])
    eli_inputs.append(company_inputs[-1])
    eli_inputs.append(experience_inputs[-1])
    print("Inputs -", eli_inputs)
    return eli_inputs


def calculateAge(birthDate):
    today = date.today()
    age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
    print("Age: ", age)
    return age


df_company = pd.read_csv('data/Company_list.csv')
def company_rank(name):
    name = name.lower()
    if name in df_company['Company_name']:
        rank = df_company['Rank']
        return rank
    else:
        rank = 3
        return rank


df_name = pd.read_csv('data/Indian-Male-Names.csv')
def gender_code(name):
    name.lower()
    name = name.split(' ')
    if any(name[0]) in df_name['name']:
        gen = 1
    else:
        gen = 0
    return gen


def inputs_processing(user):
    global savings
    inputs = eligibility_input(user)
    gen = inputs[1][0]
    dob = inputs[2]
    str1 = ""
    for x in dob:
        str1 += str(x)
    dobs = str1
    c_date = datetime.strptime(dobs, '%d%m%Y').strftime('%Y-%m-%d')
    db = pd.to_datetime(c_date)  # convert into datetime
    age = calculateAge(db)
    income = inputs[3][0]
    expenses = inputs[4][0]
    savings = int(income) - int(expenses)
    company_name = inputs[5][0]
    rank = company_rank(company_name)
    exp = inputs[6][0]
    exp = int(exp)
    user_input_dict = {"Age":age, "Income": int(income), "Expenses": int(expenses),
                       "Company": rank, "Experience": exp, "Gender_m": gen}

    print("Input_dict: ", user_input_dict)
    input_df = pd.DataFrame(user_input_dict, index=[0])
    print('Input_df: ', input_df)
    pred = find_eligibility(input_df)
    print("Eligibility_score: ", pred[0])
    return pred[0]


def eligibility_response(user):
    global savings
    sleep(2)
    result = inputs_processing(user)
    if result == 0:
        response = "I regret to inform you that, currently you are not eligible for the loan. " \
                   "However, you may try again after few months. Is there anything else I may help you with"
    else:
        amount = (int(savings)*12)*3
        formatted_currency = "{}".format(amount)
        response = "Congratulations!!, you are eligible for a loan, up to Rupees '\u20B9'{}. " \
                   "Would you like to apply for the loan".format(formatted_currency)
    print(response)
    return response


