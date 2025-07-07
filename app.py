from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Load and train base model
data = pd.read_csv('sentiment.csv')
x = data['Text']
y = data['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

vectorizer = TfidfVectorizer(stop_words='english')
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

base_model = SVC()
base_model.fit(x_train_vec, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        user_input_vec = vectorizer.transform([user_input])
        prediction = base_model.predict(user_input_vec)[0]
    return render_template("index.html", prediction=prediction)


@app.route("/compare")
def compare():
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train_vec, y_train)
        y_pred = model.predict(x_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = report

    accuracy = {name: report['accuracy'] for name, report in results.items()}
    precision = {name: report['macro avg']['precision'] for name, report in results.items()}
    recall = {name: report['macro avg']['recall'] for name, report in results.items()}
    f1_score = {name: report['macro avg']['f1-score'] for name, report in results.items()}

    # Plot and save
    x_labels = list(models.keys())
    x = np.arange(len(x_labels))
    width = 0.2

    plt.figure(figsize=(12, 8))
    plt.bar(x - 1.5*width, list(accuracy.values()), width, label='Accuracy')
    plt.bar(x - 0.5*width, list(precision.values()), width, label='Precision')
    plt.bar(x + 0.5*width, list(recall.values()), width, label='Recall')
    plt.bar(x + 1.5*width, list(f1_score.values()), width, label='F1-Score')

    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.tight_layout()

    # Save to static
    # plt.savefig('static/comparison_plot.png')
    plt.close()

    return render_template(
        "comparison.html",
        models=x_labels,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )


if __name__ == '__main__':
    app.run(debug=True)
