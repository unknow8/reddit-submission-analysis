# CMPT353-Project

## Gather and clean data
### extract_subsets.py
run ```python spark-submit extract_subsets.py ```  on the cluster
output 'submissions'

### transfer_to_csv.py
run ```python python transfer_to_csv.py ```
input 'submissions', output 'csv/submissions.csv'

### filter.py
run ```python python filter.py ```
input 'csv/submissions.csv', output output 'csv/filtered_submissions.csv' 

## Analyze data
### time_analysis.py
run ```python python time_analysis.py ```
input 'csv/filtered_submissions.csv', output images 'output/timeGoodBad/' 'output/subreddits_comments_per_hour/' 'output/totaltotal_comments_per_hour/'
text files 'csv/text_contents'

### sentiment_analysis.py
run ```python python sentiment_analysis.py ```
input 'csv/filtered_submissions.csv', output image 'output/sentiment/hist.png'

### readability_analysis.py
run ```python python readability_analysis.py ```
input 'csv/filtered_submissions.csv', output image 'output/readability/hist.png'

## Libraries
### We used TextBlob library https://textblob.readthedocs.io/en/dev/ for text sentiment
```python pip install textblob  to download the library ```

### We used py-readability-metrics https://pypi.org/project/py-readability-metrics/#flesch-kincaid-grade-level for text readability
```python pip install py-readability-metrics ```

### We also used pandas, numpy, matplotlib, stats, seaborn
