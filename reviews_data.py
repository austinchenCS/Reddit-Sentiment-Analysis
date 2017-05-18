import csv
from operator import itemgetter

import pandas as pd
import numpy as np

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer

# This uses ~3000 reviews from Amazon, Yelp, and IMDB.
def reviews_dataset():
    training_data = pd.read_csv(open("datasets/reviews_data.csv", 'rU'), encoding='utf-8', engine='c')
    training_data['polarity_num'] = training_data.polarity.map({'0': int(0), '1': int(1)})

    X = training_data.sentence
    X = Imputer().fit_transform(X)
    y = training_data.polarity_num

    # now, we vectorize this dataset.
    vect = CountVectorizer()

    reddit_comments = pd.read_csv("top_reddit_comments.csv")
    XX = reddit_comments.comments


    X_dtm = vect.fit_transform(X)
    XX_dtm = vect.transform(XX)

    # # Evaluate the data with the naive bayes model
    nb = MultinomialNB()

    # naive bayes with training data and training results
    nb.fit(X_dtm, y)

    y_pred_class = nb.predict(XX_dtm)

    all_polarity = []

    for i in y_pred_class:
        all_polarity.append(int(i))

    all_polarity_english = []

    # Convert all sentiments values to english sentiments
    for sentiment in all_polarity:
        if sentiment is 0:
            all_polarity_english.append("negative")
        elif sentiment is 1:
            all_polarity_english.append("positive")


    # Read from coments generate CSV results
    all_comments = []

    with open('top_reddit_comments.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            all_comments.append(row[0])

    all_comments.pop(0)

    # Write all comments to a .csv file for analysis
    with open('reddit_comments_results.csv', 'wb') as my_csv_file:
        the_data_writer = csv.writer(my_csv_file, delimiter=',')
        label = ("comments", "sentiment", "sentiment_num")
        the_data_writer.writerow(label)
        counter = 0
        for comment in all_comments:
            row_temp = (comment, all_polarity_english[counter], all_polarity[counter])
            the_data_writer.writerow(row_temp)
            counter += 1

    print ("Positivity/Negativity of Reddit: ")

    polarity_counter = {}
    for polarity in all_polarity_english:
        if sentiment in polarity_counter:
            polarity_counter[sentiment] += 1
        else:
            polarity_counter[sentiment] = 1

    sentiments_pct = {}

    for key in polarity_counter:
        sentiments_pct[key] = float(float(polarity[key]) / float(len(all_polarity_english)))

    for key, value in sorted(sentiments_pct.items(), key=itemgetter(1), reverse=True):
        print("   " + str(key) + ": " + str(value * 100) + "%")

