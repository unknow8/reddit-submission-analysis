import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv('csv/filtered_submissions.csv')

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour_of_day'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.day_name()

total_agg_data = data.groupby(['hour_of_day', 'day_of_week']).agg(
    all_total_comments=('num_comments', 'sum'),
    all_num_submissions=('num_comments', 'count')
).reset_index()

agg_data = data.groupby(['subreddit', 'hour_of_day', 'day_of_week']).agg(
    total_comments=('num_comments', 'sum'),
    num_submissions=('num_comments', 'count')
).reset_index()

agg_data['avg_comments_per_submission'] = agg_data['total_comments'] / agg_data['num_submissions']

output_dir_total = 'output/total_comments_per_hour'
output_dir_subreddits = 'output/subreddits_comments_per_hour'
output_dir_text = 'output/text_contents'
os.makedirs(output_dir_total, exist_ok=True)
os.makedirs(output_dir_subreddits, exist_ok=True)
os.makedirs(output_dir_text, exist_ok=True)

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days_palette = sns.color_palette('husl', n_colors=len(days_order))

# small multiple time series: https://seaborn.pydata.org/examples/timeseries_facets.html
plt.figure(figsize=(12, 8))
sns.set(style='whitegrid', palette='husl')
g = sns.FacetGrid(total_agg_data, col='day_of_week', col_wrap=4, height=4, col_order=days_order, palette=days_palette)
g.map(sns.lineplot, 'hour_of_day', 'all_total_comments', marker='o', errorbar=None)
g.set_axis_labels('Hour of Day', 'Num Comments / Avg per Submission')
plt.tight_layout()
total_comments_per_hour_plot = os.path.join(output_dir_total, 'total_comments_per_hour.png')
plt.savefig(total_comments_per_hour_plot)
plt.close()

total_text_file = os.path.join(output_dir_text, 'total_comments_per_hour.txt')
with open(total_text_file, 'w') as file:
    file.write(f"Contents of {total_comments_per_hour_plot}:\n")
    file.write("Total Comments per Hour and Avg Comments per Submission for Each Day of the Week\n\n")
    for day in days_order:
        file.write(f"Day: {day}\n")
        day_data = agg_data[agg_data['day_of_week'] == day]
        for index, row in day_data.iterrows():
            file.write(f"Hour: {row['hour_of_day']} - Total Comments: {row['total_comments']} - Avg Comments per Submission: {row['avg_comments_per_submission']:.2f}\n")
        file.write("\n")

for subreddit, subreddit_data in agg_data.groupby('subreddit'):
    plt.figure(figsize=(12, 8))
    g = sns.FacetGrid(subreddit_data, col='day_of_week', col_wrap=4, height=4, col_order=days_order, palette=days_palette)
    g.map(sns.lineplot, 'hour_of_day', 'total_comments', marker='o', errorbar=None)
    g.map(sns.lineplot, 'hour_of_day', 'avg_comments_per_submission', marker='x', errorbar=None)
    g.set_axis_labels('Hour of Day', f'Total / Avg Num Comments - {subreddit}')
    plt.tight_layout()
    subreddit_comments_per_hour_plot = os.path.join(output_dir_subreddits, f'{subreddit}_comments_per_hour.png')
    plt.savefig(subreddit_comments_per_hour_plot)
    plt.close()

    subreddit_avg_comments = subreddit_data['avg_comments_per_submission'].mean()
    subreddit_text_file = os.path.join(output_dir_text, f'{subreddit}_comments_per_hour.txt')
    with open(subreddit_text_file, 'w') as file:
        file.write(f"Contents of {subreddit_comments_per_hour_plot}:\n")
        file.write(f"Total Comments per Hour and Avg Comments - {subreddit}\n\n")
        for day in days_order:
            file.write(f"Day: {day}\n")
            day_data = subreddit_data[subreddit_data['day_of_week'] == day]
            for index, row in day_data.iterrows():
                file.write(f"Hour: {row['hour_of_day']} - Total Comments: {row['total_comments']} - Avg Comments per Submission: {row['avg_comments_per_submission']:.2f}\n")
            file.write("\n")
        file.write(f"Total Average Comments per Submission: {subreddit_avg_comments:.2f}\n")

# ---------analysis starts here--------

good_submissions = data[data['good']]  
bad_submissions = data[~data['good']] 

avg_comments_per_hour = agg_data.groupby('hour_of_day')['avg_comments_per_submission'].mean()

good_above_avg = good_submissions.groupby(['hour_of_day', 'day_of_week', 'subreddit'])['num_comments'].apply(lambda x: (x >= avg_comments_per_hour[x.name[0]]).mean() * 100).reset_index(name='percent_good_above_avg')

bad_above_avg = bad_submissions.groupby(['hour_of_day', 'day_of_week', 'subreddit'])['num_comments'].apply(lambda x: (x >= avg_comments_per_hour[x.name[0]]).mean() * 100).reset_index(name='percent_bad_above_avg')

output_dir = 'output/timeGoodBad'
os.makedirs(output_dir, exist_ok=True)

time_good_bad_output = os.path.join(output_dir, 'allResults.txt')
with open(time_good_bad_output, 'w') as file:
    for subreddit, subreddit_data in good_above_avg.groupby('subreddit'):
        file.write(f"Subreddit: {subreddit}\n")
        for day, day_data in subreddit_data.groupby('day_of_week'):
            file.write(f"Day: {day}\n")
            file.write(day_data.to_string(index=False) + '\n\n')
    
    for subreddit, subreddit_data in bad_above_avg.groupby('subreddit'):
        file.write(f"Subreddit: {subreddit}\n")
        for day, day_data in subreddit_data.groupby('day_of_week'):
            file.write(f"Day: {day}\n")
            file.write(day_data.to_string(index=False) + '\n\n')

# worst vs bad results
margin = 0.1

def find_best_worst_hours(df, col_name):
    max_val = df[col_name].max()
    min_val = df[col_name].min()
    max_margin = (max_val) * margin
    min_margin = (min_val) * margin

    best_hours = df[df[col_name] >= (max_val - max_margin)]
    worst_hours = df[df[col_name] <= (min_val + min_margin)]
    return best_hours, worst_hours

best_hours, worst_hours = [], []

for subreddit, subreddit_data in good_above_avg.groupby('subreddit'):
    for day, day_data in subreddit_data.groupby('day_of_week'):
        total_count = day_data['percent_good_above_avg'].count()
        best, worst = find_best_worst_hours(day_data, 'percent_good_above_avg')
        best_hours.append(best.assign(Percentage=f"{len(best)/total_count:.2%}"))
        worst_hours.append(worst.assign(Percentage=f"{len(worst)/total_count:.2%}"))

best_hours_df = pd.concat(best_hours)
worst_hours_df = pd.concat(worst_hours)

time_good_bad_results_output = os.path.join(output_dir,'BestWorstResults.txt')
with open(time_good_bad_results_output, 'w') as file:
    for subreddit, subreddit_data in best_hours_df.groupby('subreddit'):
        file.write(f"Subreddit: {subreddit}\n")
        for day, day_data in subreddit_data.groupby('day_of_week'):
            file.write(f"Best hour for {day}: {day_data['hour_of_day'].values}\n")
    
    for subreddit, subreddit_data in worst_hours_df.groupby('subreddit'):
        file.write(f"Subreddit: {subreddit}\n")
        for day, day_data in subreddit_data.groupby('day_of_week'):
            file.write(f"Worst hour for {day}: {day_data['hour_of_day'].values}\n")

# HeatMap: https://seaborn.pydata.org/examples/structured_heatmap.html

good_submissions = data[data['good']]
bad_submissions = data[~data['good']]

heatmap_output_dir = 'output/timeGoodBad'
os.makedirs(heatmap_output_dir, exist_ok=True)

for df in [good_above_avg, bad_above_avg]:
    for subreddit, subreddit_data in df.groupby('subreddit'):
        all_days_heatmap_data = subreddit_data.pivot_table(index='hour_of_day', columns='day_of_week', values='percent_good_above_avg')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(all_days_heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", linewidths=.5)
        plt.title(f"{subreddit} - Best Hours (All Days)" if df.equals(good_above_avg) else f"{subreddit} - Worst Hours (All Days)")
        plt.xlabel('Day of the Week')
        plt.ylabel('Hour of the Day')

        heatmap_filename = f"{subreddit}_{'Best' if df.equals(good_above_avg) else 'Worst'}_AllDays_heatmap.png"
        heatmap_filepath = os.path.join(heatmap_output_dir, heatmap_filename)
        plt.savefig(heatmap_filepath)
        plt.close()
