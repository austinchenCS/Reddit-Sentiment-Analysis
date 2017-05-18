import json
import csv

counter = 0

tweets = []

with open('datasets/twitter_nltk/positive_tweets.json', 'r') as fh:
    for line in fh:
        tweet = json.loads(line)
        tweet = str(tweet['text'].encode('utf-8'))
        tweets.append(tweet)

with open('datasets/twitter_nltk/negative_tweets.json', 'r') as fh:
    for line in fh:
        tweet = json.loads(line)
        tweet = str(tweet['text'].encode('utf-8'))
        tweets.append(tweet)



# Write all comments to a .csv file for analysis
with open('tweet_data.csv', 'wb') as my_csv_file:
    the_data_writer = csv.writer(my_csv_file, delimiter=',')
    label = ("tweet", "polarity")
    the_data_writer.writerow(label)

    counter = 0
    flip = False
    for tweet in tweets:
        if counter == 5000:
            flip = True
        if flip is False:
            row_temp = (str(tweet), "positive")
        if flip is True:
            row_temp = (str(tweet), "negative")

        the_data_writer.writerow(row_temp)
        counter += 1
