# Austin Chen
# CSE 3353 - Fundamentals of Algorithms
# Final Project Spring 2017

from __future__ import print_function

# Import PRAW for reddit python wrapper
import praw
from praw.models import MoreComments

# Import CSV for .csv creation
import csv

# Import demos from other files
from iris_demo import run_iris_demo
from twitter_sentiment_demo import run_twitter_demo
from spamham_demo import run_spamham_demo

from twitter_sentiment_data import twitter_sentiment_dataset
from reviews_data import reviews_dataset

# PRAW Authentication
my_client_id = 'EgDV9NAjBOsPaA'
my_client_secret = 'Wz6sDT9BextdU29zYcRQuDDoChs'
my_user_agent = 'macOS:twitter_sentiment_analysis:v1.0.0 (by /u/rsa_tool)'
my_username = 'rsa_tool'
my_password = '@ustin_1s_c00l'

reddit = praw.Reddit(client_id = my_client_id,
                     client_secret = my_client_secret,
                     user_agent = my_user_agent,
                     username = my_username,
                     password = my_password)

# Generates a csv of the top 10 comments from the top X posts of the day
def generate_comment_data():

    # Set number of posts and timeframe to grab posts from (r/all)
    num_posts = 5
    comments_per_thread = 10
    # Time can be day, week, month, year, all
    timeframe = "day"


    top_posts_urls = []
    top_posts_ids = []

    # Get the top posts in r/all
    for submission in reddit.subreddit('all').top(timeframe, limit=num_posts):
        top_posts_urls.append(submission.title)
        top_posts_ids.append(submission.id)

    # List of all the comments
    all_comments = []
    num_top_level_comments = 0

    print('Grabbing comments from the top Reddit posts of the ' + timeframe + '...', end='')

    # Grab all comments from comment posts
    for post in top_posts_ids:
        submission = reddit.submission(id=post)

        thread_comments = 0

        for top_level_comment in submission.comments:
            if thread_comments is comments_per_thread:
                break;
            if isinstance(top_level_comment, MoreComments):
                continue
            all_comments.append((str(top_level_comment.body.encode('ascii', errors='ignore')).replace('\n', ''), ' '))
            num_top_level_comments += 1
            thread_comments += 1
            print('.', end='')

    print("\n")

    # Write all comments to a .csv file for analysis
    with open('top_reddit_comments.csv', 'wb') as my_csv_file:
        the_data_writer = csv.writer(my_csv_file, delimiter=',')
        label = ("comments", " ")
        the_data_writer.writerow(label)
        for comment in all_comments:
            the_data_writer.writerow(comment)


# Flags to run demos
RUN_IRIS = True
RUN_TWITTER_DEMO = True
RUN_SPAMHAM_DEMO = True

# Main function
def main():

    if RUN_IRIS is True: run_iris_demo()
    if RUN_TWITTER_DEMO is True: run_twitter_demo()
    if RUN_SPAMHAM_DEMO is True: run_spamham_demo()

    # Grab the comments from the top posts on reddit and create a .csv file of each comment
    generate_comment_data()

    twitter_sentiment_dataset()


# Call main
if __name__ == "__main__":
    main()