from flask import Flask, render_template, request, redirect, url_for, flash
from threading import Thread
import time
from textblob import TextBlob
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from langdetect import detect
from googletrans import Translator

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flashing messages

translator = Translator()

# Load feedback data
feedback_data = pd.DataFrame(columns=['feedback', 'sentiment'])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    data = get_dashboard_data()  # Retrieve data to display on the dashboard
    templates = get_templates()  # Retrieve pre-built templates
    return render_template('dashboard.html', data=data, templates=templates)

# Survey route
@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        feedback = request.form['feedback']
        language = detect(feedback)
        if language != 'en':
            feedback = translate_text(feedback, src=language, dest='en')
        sentiment = analyze_sentiment(feedback)
        save_feedback(feedback, sentiment)
        flash('Survey successfully submitted!')
        return redirect(url_for('home'))
    return render_template('survey.html')

# Alerts route
@app.route('/alerts')
def alerts():
    alerts = check_for_alerts()
    return render_template('alerts.html', alerts=alerts)

# Background data processing
def process_data():
    while True:
        update_dashboard_data()
        time.sleep(10)

@app.before_first_request
def activate_job():
    thread = Thread(target=process_data)
    thread.start()

# Sentiment analysis function
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    return analysis.sentiment.polarity

# Save feedback and sentiment to the dataset
def save_feedback(feedback, sentiment):
    global feedback_data
    feedback_data = feedback_data.append({'feedback': feedback, 'sentiment': sentiment}, ignore_index=True)
    feedback_data.to_csv('data/feedback.csv', index=False)

# Get dashboard data
def get_dashboard_data():
    global feedback_data
    positive_feedback = feedback_data[feedback_data['sentiment'] > 0].shape[0]
    negative_feedback = feedback_data[feedback_data['sentiment'] < 0].shape[0]
    return {
        'positive_feedback': positive_feedback,
        'negative_feedback': negative_feedback,
        'feedback_over_time': feedback_data[['sentiment']].to_dict(orient='records')
    }

# Update dashboard data with predictive analytics
def update_dashboard_data():
    global feedback_data
    if feedback_data.shape[0] > 10:  # Ensure we have enough data for prediction
        X = feedback_data['sentiment'].values.reshape(-1, 1)
        y = np.arange(len(feedback_data))  # Simulated time series data for example
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor()
        model.fit(X_scaled, y)
        future_sentiment = model.predict(scaler.transform(np.array([[0.0]])))  # Predict for neutral sentiment
        feedback_data['predicted_sentiment'] = model.predict(X_scaled)
        print("Future Sentiment Prediction:", future_sentiment)

# Alert checking function
def check_for_alerts():
    global feedback_data
    alerts = []
    if feedback_data[feedback_data['sentiment'] < -0.5].shape[0] > 5:  # Example threshold
        alerts.append('High number of negative feedback detected!')
    return alerts

# Translate text to English
def translate_text(text, src='auto', dest='en'):
    translation = translator.translate(text, src=src, dest=dest)
    return translation.text

# Pre-built templates
def get_templates():
    return [
        {
            'name': 'Sales Dashboard',
            'widgets': ['Sales Over Time', 'Revenue by Product', 'Top Customers']
        },
        {
            'name': 'Customer Feedback Dashboard',
            'widgets': ['Feedback Sentiment Over Time', 'Positive vs. Negative Feedback', 'Top Keywords']
        }
    ]

if __name__ == '__main__':
    app.run(debug=True)

