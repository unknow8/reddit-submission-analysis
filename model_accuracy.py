from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('csv/filtered_submissions.csv')  # Replace with your CSV file

numerical_features = ['created_utc', 'downs', 'gilded', 'num_comments', 'score', 'ups']
X = data[numerical_features]
y = data['subreddit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'MLP': MLPClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30),
    'Naive Bayes': GaussianNB(var_smoothing=1e-9),
    'SVC': SVC()
}

reports = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    reports[name] = report

with open('classification_reports.txt', 'w') as file:
    for name, report_dict in reports.items():
        file.write(f"Model: {name}\n")
        file.write("Classification Report:\n")
        file.write(f"{pd.DataFrame(report_dict).transpose()}\n")
        file.write("----------------------------\n")
