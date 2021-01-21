def predict(review):
    model = pickle.load(open('model_hotelreviews.pkl', 'rb'))
    l = []
    l.append(review)
    sent = model.predict(l)
    if sent[0]== 0:
        return "Negative"
    else:
        return "Positive"
       
       
customer_reviews = ['Siddharth from MyLoanCare helped well and the loan got processed fast','Its a very good experience with you. Here I found the best deal and best interest rate of all banks'
                   ,'Very good. Special thanks to Saurabh Gandhi from MyLoanCare and Ram from HDFC Bank',
                   'Amazing service. Really happy with the team. Definitely recommend it to everyone.',
                   'Worst service by the bank',
                   'had very bad expierience']
for i in customer_reviews:
    print(predict(i))
