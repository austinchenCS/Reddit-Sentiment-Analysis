import pandas as pd

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def run_twitter_demo():
    data = pd.read_csv("datasets/tweet_data_2.csv")
    data['polarity_num'] = data.polarity.map({'negative': 0, 'positive': 1})

    X = data.tweet
    y = data.polarity_num


    #print(data.head())

    #print("shape of all sentences:                " + str(X.shape))
    #print("shape of all sentence classifications: " + str(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #print("")
    #print("shape of all sentences used for model training:                 " + str(X_train.shape))
    #print("shape of all sentence classifications used for model testing:   " + str(y_train.shape))
    #print("shape of all sentences used for TESTING:                        " + str(X_test.shape))
    #print("shape of all sentence classifications used for TESTING:         " + str(y_test.shape))

    # now, we vectorize this dataset.
    vect = CountVectorizer()

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)


    # # Evaluate the data with the naive bayes model
    nb = MultinomialNB()

    # naive bayes with training data and training results
    nb.fit(X_train_dtm, y_train)

    y_pred_class = nb.predict(X_test_dtm)

    #print("")
    #print("Naive Bayes Classifier - Percent accuracy of Twitter model testing: " + str(100 * metrics.accuracy_score(y_test, y_pred_class)) + "%")

    # Evaluate the model with the logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

    #print("Logistic Regression - Percent accuracy of Twitter model testing: " + str(100 * metrics.accuracy_score(y_test, y_pred_class)) + "%")
    #print("")

    return 100 * metrics.accuracy_score(y_test, y_pred_class)

average = 0
for i in range (0, 10):
    print(i)
    average += run_twitter_demo()

average = average/10
print(average)