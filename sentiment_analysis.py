import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
from scipy import stats
import numpy as np

df = pd.read_csv('csv/filtered_submissions.csv')

# remove entries with no selftext, or [deleted] or [removed]
df = df[df['selftext'].isna() == False]
df = df[df['selftext'] != '[deleted]']
df = df[df['selftext'] != '[removed]']

# calculate sentiment polarity score for each entry
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
df['sentiment'] = df['selftext'].apply(calculate_sentiment)

# split entries into good submissions and bad submissions
good_submission = df[df['good']]
bad_submission = df[df['good'] == False]

x1 = good_submission['sentiment']
x2 = bad_submission['sentiment']

# tested transform data, didn't work
# x1 = np.log(good_submission['sentiment'] + 1)
# x2 = np.log(bad_submission['sentiment'] + 1)

# x1 = (x1 - x1.mean())/x1.std()
# x2 = (x2 - x2.mean())/x2.std()

# plot histogram for sentiment polarity score for both good and bad submission
# for visual comparison
plt.figure()
plt.hist(x1, alpha=0.5, bins=100, label='good')
plt.hist(x2, alpha=0.5, bins=100, label='bad')
plt.legend()
plt.title('Sentiment score histograms')
plt.xlabel('sentiment score')
plt.ylabel('count')
os.makedirs('output/sentiment', exist_ok=True)
plt.savefig('output/sentiment/hist.png')

# prepare for T-test
# testing normality
print('For sentiment polarity score: ')
print('Normality p-value for good submissions : ', stats.normaltest(x1).pvalue)
print('Normality p-value for bad submissions: ',stats.normaltest(x2).pvalue)

print('Kurtosis for good submissions : ', stats.kurtosis(x1))
print('Kurtosis for bad submissions : ', stats.kurtosis(x2))

# testing for equal variance
print('Equal variance p-value for good vs bad submissions : ',stats.levene(x1, x2).pvalue)

# T-test
print('T-test p-value for good vs bad submissions: ',stats.ttest_ind(x1, x2).pvalue)

print('Good submission mean: ', x1.mean(), 'Bad submission mean: ', x2.mean())
