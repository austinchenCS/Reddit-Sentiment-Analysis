import csv
from operator import itemgetter

import pandas as pd

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def twitter_sentiment_dataset():
    training_data = pd.read_csv("datasets/twitter_13_sentiments.csv")
    training_data['sentiment_num'] = training_data.sentiment.map({'empty': 0, 'sadness': 1, 'worry': 2, 'hate': 3,
                                                                  'anger': 4, 'neutral': 5, 'happiness': 6,
                                                                  'surprise': 7, 'boredom': 8, 'relief': 9,
                                                                  'enthusiasm': 10, 'love': 11, 'fun': 12})

    # Create matrices with data
    X = training_data.content
    y = training_data.sentiment_num

    # now, we vectorize this dataset.
    vect = CountVectorizer()

    reddit_comments = pd.read_csv("top_reddit_comments.csv")
    XX = reddit_comments.comments

    # Transform the matrices into document-term matrices
    X_dtm = vect.fit_transform(X)
    XX_dtm = vect.transform(XX)

    # # Evaluate the data with the naive bayes model
    nb = MultinomialNB()

    # naive bayes with training data and training results
    nb.fit(X_dtm, y)

    # PREDICT THE RESULTS OF DATA (REDDIT COMMENTS)
    y_pred_class = nb.predict(XX_dtm)

    all_sentiments = []

    for i in y_pred_class:
        all_sentiments.append(int(i))

    all_sentiments_english = []

    # Convert all sentiments values to english sentiments
    for sentiment in all_sentiments:
        if sentiment is 0:
            all_sentiments_english.append("empty")
        elif sentiment is 1:
            all_sentiments_english.append("sadness")
        elif sentiment is 2:
            all_sentiments_english.append("worry")
        elif sentiment is 3:
            all_sentiments_english.append("hate")
        elif sentiment is 4:
            all_sentiments_english.append("anger")
        elif sentiment is 5:
            all_sentiments_english.append("neutral")
        elif sentiment is 6:
            all_sentiments_english.append("happiness")
        elif sentiment is 7:
            all_sentiments_english.append("surprise")
        elif sentiment is 8:
            all_sentiments_english.append("boredom")
        elif sentiment is 9:
            all_sentiments_english.append("relief")
        elif sentiment is 10:
            all_sentiments_english.append("enthusiasm")
        elif sentiment is 11:
            all_sentiments_english.append("love")
        elif sentiment is 12:
            all_sentiments_english.append("fun")


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
            row_temp = (comment, all_sentiments_english[counter], all_sentiments[counter])
            the_data_writer.writerow(row_temp)
            counter += 1

    print ("Emotions of Reddit: ")

    sentiment_counter = {}
    for sentiment in all_sentiments_english:
        if sentiment in sentiment_counter:
            sentiment_counter[sentiment] += 1
        else:
            sentiment_counter[sentiment] = 1

    sentiments_pct = {}

    for key in sentiment_counter:
        sentiments_pct[key] = float(float(sentiment_counter[key]) / float(len(all_sentiments_english)))

    for key, value in sorted(sentiments_pct.items(), key=itemgetter(1), reverse=True):
        print("   " + str(key) + ": " + str(value * 100) + "%")

