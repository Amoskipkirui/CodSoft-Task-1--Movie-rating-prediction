from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('rf_model.pkl')

# Define the top 3 important features
top_features = ['Duration(minutes)', 'Year', 'Votes']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract the input data from the form
    duration = float(request.form.get('Duration(minutes)'))
    year = float(request.form.get('Year'))
    votes = float(request.form.get('Votes'))

    # Create a DataFrame with the selected top features
    input_data = pd.DataFrame({
        'Duration(minutes)': [duration],
        'Year': [year],
        'Votes': [votes]
    })

    # Perform prediction using the model
    prediction = model.predict(input_data)[0]

    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
