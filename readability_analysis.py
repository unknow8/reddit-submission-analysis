import pandas as pd
from readability import Readability
import matplotlib.pyplot as plt
import os
from scipy import stats
import numpy as np
# in case of punkt missing warning
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

df = pd.read_csv('csv/filtered_submissions.csv')

# remove entries with no selftext, or [deleted] or [removed]
df = df[df['selftext'].isna() == False]
df = df[df['selftext'] != '[deleted]']
df = df[df['selftext'] != '[removed]']

# remove entries with selftext less than 100 words, the Readability() need
# at least 100 words for calcualtion
# the reason the code is >= 115 is split() define word differently than readability
# the extra 15 act as a buffer for Readability() to work
def hundred_words(text):
    return len(text.split()) >= 115
df['more_than_100'] = df['selftext'].apply(hundred_words)
df = df[df['more_than_100']]

# calculate flesch reading ease score for each entry
def calculate_readability(text):
    r = Readability(text)
    return r.flesch().score
df['readability'] = df['selftext'].apply(calculate_readability)

# # for testing, save the df locally
# # df.to_csv('csv/tmp.csv', index=False)
# df = pd.read_csv('csv/tmp.csv')

# split entries into good submissions and bad submissions
good_submission = df[df['good']]
bad_submission = df[df['good'] == False]

x1 = good_submission['readability']
x2 = bad_submission['readability']

print('Good submission mean: ', x1.mean(), 'Bad submission mean: ', x2.mean())

# data transformation
x1 = x1[x1 >= 0]
x2 = x2[x2 >= 0]

x1 = x1**3
x2 = x2**3

# plot histogram for readability score for both good and bad submission
# for visual comparison
plt.figure()
plt.hist(x1, alpha=0.5, bins=100, label='good')
plt.hist(x2, alpha=0.5, bins=100, label='bad')
plt.legend()
plt.title('Flesch reading ease score histograms')
plt.xlabel('Flesch reading ease score')
plt.ylabel('count')
os.makedirs('output/readability', exist_ok=True)
plt.savefig('output/readability/hist.png')


# prepare for T-test
# testing normality
print('For readability score: ')
print('Normality p-value for good submissions : ', stats.normaltest(x1).pvalue)
print('Normality p-value for bad submissions: ',stats.normaltest(x2).pvalue)

print('Kurtosis for good submissions : ', stats.kurtosis(x1))
print('Kurtosis for bad submissions : ', stats.kurtosis(x2))

# testing for equal variance
print('Equal variance p-value for good vs bad submissions : ',stats.levene(x1, x2).pvalue)

# T-test
print('T-test p-value for good vs bad submissions: ',stats.ttest_ind(x1, x2).pvalue)



