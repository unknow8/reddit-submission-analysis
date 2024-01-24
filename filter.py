import pandas as pd

# Load the original data from your CSV file
data = pd.read_csv('csv/submissions.csv')  # Replace with your CSV file
# data1 = data

# # there is no rows with null value
# # Remove rows where 'created_utc' is null
# data = data.dropna(subset=['created_utc'])
# print(data1.compare(data))

# # no need to replace, there is no null values, checked with .compare
# # Replace null values in other numerical columns with 0
# numerical_columns = ['downs', 'gilded', 'num_comments', 'score', 'ups']
# data[numerical_columns] = data[numerical_columns].fillna(0)
# print(data1.compare(data))

# Convert 'created_utc' to a datetime format and create a new 'timestamp' column
data['timestamp'] = pd.to_datetime(data['created_utc'], unit='s')

data['is_media'] = data['media'].isna() == False

data = data.drop(['archived', 'author', 'author_flair_css_class', 
                  'author_flair_text', 'created_utc',  'domain', 'downs', 'edited', 'gilded',
                  'hide_score', 'id', 'link_flair_css_class', 'link_flair_text', 'name', 'over_18',
                  'permalink', 'quarantine', 'retrieved_on', 'saved', 'media', 'secure_media', 'stickied',
                  'subreddit_id', 'thumbnail', 'ups', 'url'
                  ], axis=1)

# calculate the 50th quantile score for each subreddits
tmp = data[['subreddit', 'score']]
tmp = tmp.groupby('subreddit').quantile(0.50)
# print(tmp)

# join dataframes and create new column 'good'
data = data.merge(tmp, on='subreddit')
data['good'] = data['score_x'] >= data['score_y']
data = data.rename(columns={"score_x": "score"})
data = data.drop('score_y', axis=1)
# print(data.columns)


# Save the filtered data to a new CSV file called 'filtered_submissions.csv'
data.to_csv('csv/filtered_submissions.csv', index=False)
