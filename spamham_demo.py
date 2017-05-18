import pandas as pd

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def run_spamham_demo():

    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    sms = pd.read_table(url, header=None, names=['label', 'message'])

    # Give all ham's 0's and spam's 1's. Because you need numbers i think.
    sms['label_num'] = sms.label.map({'ham': 0, 'spam': 1})

    print(sms.head())

    X = sms.message
    y = sms.label_num

    print("All data: \n")
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("X_train: " + str(X_train.shape))
    print("X_test: " + str(X_test.shape))
    print("y_train: " + str(y_train.shape))
    print("y_test: " + str(y_test.shape))

    # Create our count vectorizer object.
    vect = CountVectorizer()

    X_train_dtm = vect.fit_transform(X_train)
    # to do both fitting and transforming in one sweep
    # Oh its also faster and it's what you do in the real world

    X_test_dtm = vect.transform(X_test)

    nb = MultinomialNB()

    nb.fit(X_train_dtm, y_train)

    y_pred_class = nb.predict(X_test_dtm)

    print("")
    print("Accuracy of spam vs. ham testing: " + str(100 * metrics.accuracy_score(y_test, y_pred_class)) + "%")