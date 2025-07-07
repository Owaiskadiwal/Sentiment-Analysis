from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis pipelines
sentiment_pipeline_1 = pipeline('sentiment-analysis', model='avichr/heBERT_sentiment_analysis')
sentiment_pipeline_2 = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
sentiment_pipeline_3 = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Get predictions from all models
    results_1 = sentiment_pipeline_1(text)
    results_2 = sentiment_pipeline_2(text)
    results_3 = sentiment_pipeline_3(text)
    
    # Extract labels and scores
    labels = [result['label'] for result in [results_1[0], results_2[0], results_3[0]]]
    scores = [result['score'] for result in [results_1[0], results_2[0], results_3[0]]]
    
    # Map labels to numerical values
    label_to_value = {
        'LABEL_0': -1,  # Adjust based on your model's label meanings
        'LABEL_1': 1,
        'negative': -1,  # Example for models with different labels
        'positive': 1,
        'neutral': 0
    }
    
    # Convert labels to numerical values
    values = [label_to_value.get(label, 0) for label in labels]
    
    # Majority sentiment determination
    from collections import Counter
    sentiment_counter = Counter(labels)
    majority_sentiment = sentiment_counter.most_common(1)[0][0]
    majority_sentiment_value = label_to_value.get(majority_sentiment, 0)
    
    # Calculate average sentiment score
    average_score = sum(scores) / len(scores) if scores else 0
    average_sentiment = round(average_score, 2) if scores else 'N/A'
    
    return render_template('index.html', prediction={
        'heBERT_sentiment_analysis': results_1,
        'twitter_roberta_base_sentiment': results_2,
        'twitter_xlm_roberta_base_sentiment': results_3
    }, majority_sentiment=majority_sentiment, average_sentiment=average_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
