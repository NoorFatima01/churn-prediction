from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the pickled file
model = pickle.load(open('model.sav', 'rb'))

# Route to display the form
@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is the file you shared earlier

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting form data
    features = [
        int(request.form['SeniorCitizen']),
        float(request.form['MonthlyCharges']),
        float(request.form['TotalCharges']),
        int(request.form['Churn']),
        int(request.form['gender_Male']),
        int(request.form['Partner_Yes']),
        int(request.form['Dependents_Yes']),
        int(request.form['PhoneService_Yes']),
        int(request.form['MultipleLines_No phone service']),
        int(request.form['MultipleLines_Yes']),
        int(request.form['InternetService_Fiber optic']),
        int(request.form['InternetService_No']),
        int(request.form['OnlineSecurity_No internet service']),
        int(request.form['OnlineSecurity_Yes']),
        int(request.form['OnlineBackup_No internet service']),
        int(request.form['OnlineBackup_Yes']),
        int(request.form['DeviceProtection_No internet service']),
        int(request.form['DeviceProtection_Yes']),
        int(request.form['TechSupport_No internet service']),
        int(request.form['TechSupport_Yes']),
        int(request.form['StreamingTV_No internet service']),
        int(request.form['StreamingTV_Yes']),
        int(request.form['StreamingMovies_No internet service']),
        int(request.form['StreamingMovies_Yes']),
        int(request.form['Contract_One year']),
        int(request.form['Contract_Two year']),
        int(request.form['PaperlessBilling_Yes']),
        int(request.form['PaymentMethod_Credit card (automatic)']),
        int(request.form['PaymentMethod_Electronic check']),
        int(request.form['PaymentMethod_Mailed check']),
        int(request.form['tenure_group_13 - 24']),
        int(request.form['tenure_group_25 - 36']),
        int(request.form['tenure_group_37 - 48']),
        int(request.form['tenure_group_49 - 60']),
        int(request.form['tenure_group_61 - 72'])
    ]

    # Convert the input features to a numpy array for prediction
    features = np.array(features).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(features)

    # Interpret the prediction (assuming 0 = No Churn, 1 = Churn)
    if prediction == 1:
        prediction_text = "The customer is likely to churn."
    else:
        prediction_text = "The customer is not likely to churn."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
