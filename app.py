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
from flask.json import JSONEncoder

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flashing messages

translator = Translator()

# Custom JSON Encoder to handle Undefined values
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            return super(CustomJSONEncoder, self).default(obj)
        except TypeError:
            return str(obj)

app.json_encoder = CustomJSONEncoder

# Load feedback data
feedback_data = pd.DataFrame(columns=['feedback', 'rating', 'yesno', 'text_sentiment'])

# Survey templates
survey_templates = {
    'Customer Satisfaction': [
        {'question': 'How satisfied are you with our product?', 'type': 'rating'},
        {'question': 'Would you recommend our product to others?', 'type': 'yesno'},
        {'question': 'Any additional feedback?', 'type': 'text'}
    ],
    'Employee Feedback': [
        {'question': 'How satisfied are you with your current role?', 'type': 'rating'},
        {'question': 'Do you feel valued at work?', 'type': 'yesno'},
        {'question': 'Any suggestions for improvement?', 'type': 'text'}
    ]
}

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

# Survey creation route
@app.route('/create_survey', methods=['GET', 'POST'])
def create_survey():
    if request.method == 'POST':
        template = request.form['template']
        survey = survey_templates[template]
        return render_template('survey_form.html', survey=survey, template=template)
    return render_template('create_survey.html', templates=survey_templates.keys())

# Survey submission route
@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    feedback = ''
    rating = None
    yesno = None
    text_sentiment = None

    # Process the responses
    for key, value in request.form.items():
        if key != 'template':
            feedback += f"{key}: {value}\n"
            if key.startswith('question_') and 'rating' in key:
                rating = int(value)
            elif key.startswith('question_') and 'yesno' in key:
                yesno = 1 if value == 'yes' else -1
            elif key.startswith('question_') and 'text' in key and value.strip() != '':
                text_sentiment = analyze_sentiment(value)

    save_feedback(feedback, rating, yesno, text_sentiment)
    update_dashboard_data()  # Ensure feedback metrics update
    flash('Survey successfully submitted!')
    return redirect(url_for('dashboard'))

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
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Save feedback and sentiment to the dataset
def save_feedback(feedback, rating, yesno, text_sentiment):
    global feedback_data
    feedback_data = feedback_data.append({'feedback': feedback, 'rating': rating, 'yesno': yesno, 'text_sentiment': text_sentiment}, ignore_index=True)
    feedback_data.to_csv('data/feedback.csv', index=False)

# Get dashboard data
def get_dashboard_data():
    global feedback_data
    rating_positive = feedback_data[(feedback_data['rating'] == 4) | (feedback_data['rating'] == 5)].shape[0]
    rating_neutral = feedback_data[feedback_data['rating'] == 3].shape[0]
    rating_negative = feedback_data[(feedback_data['rating'] == 1) | (feedback_data['rating'] == 2)].shape[0]

    yesno_positive = feedback_data[feedback_data['yesno'] == 1].shape[0]
    yesno_negative = feedback_data[feedback_data['yesno'] == -1].shape[0]

    text_positive = feedback_data[feedback_data['text_sentiment'] > 0].shape[0]
    text_neutral = feedback_data[feedback_data['text_sentiment'] == 0].shape[0]
    text_negative = feedback_data[feedback_data['text_sentiment'] < 0].shape[0]

    return {
        'rating': {'positive': rating_positive, 'neutral': rating_neutral, 'negative': rating_negative},
        'yesno': {'positive': yesno_positive, 'negative': yesno_negative},
        'text': {'positive': text_positive, 'neutral': text_neutral, 'negative': text_negative}
    }

# Update dashboard data with predictive analytics
def update_dashboard_data():
    global feedback_data
    # Dummy implementation for predictive analytics, replace with actual logic if needed
    pass

# Alert checking function
def check_for_alerts():
    global feedback_data
    alerts = []
    if feedback_data[feedback_data['text_sentiment'] < -0.5].shape[0] > 5:  # Example threshold
        alerts.append('High number of negative feedback detected!')
    return alerts

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

