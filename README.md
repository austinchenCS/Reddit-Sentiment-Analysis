# Reddit-Sentiment-Analysis

My Reddit Sentiment Analysis tool is a program to gauge the general "mood" of Reddit for any given hour/day/week/year/etc.

The sentiment analysis utilizes a Naive Bayes Classifier and a Logistic Regression Model in order to classify each Reddit comment as one of thirteen sentiments. The subsequent output is then presented as a percentage share of emotions (20% neutral, 25% happiness, 34% sadness, etc.)

The training data used is comprised of 40,000 sentiment tagged tweets.


List of sentiments used:
1. empty
2. sadness
3. worry
4. hate
5. anger
6. neutral
7. happiness
8. surprise
9. boredom
10. relief
11. enthusiasm
12. love
13. fun

# External Python libraries Used:
- scikit-learn
- pandas
- praw

