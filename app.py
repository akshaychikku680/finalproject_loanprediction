from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

#Load the scaler and model from the pickle files
with open('scale.pkl', 'rb') as scalerfile:
    scaler = pickle.load(scalerfile)
    
with open('loan.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

#creating a home page
@app.route('/')
def home():
    return render_template('index.html')

#serve the HTML
@app.route('/desc')
def des():
    return render_template('desc.html')

# Serve the HTML form
@app.route('/home')
def index():
    return render_template('home.html')

# Handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict_route():
    # Extract features from the form data
    features = [
        float(request.form['income_annum']),
        float(request.form['loan_amount']),
        float(request.form['loan_term']),
        float(request.form['cibil_score']),
        float(request.form['residential_assets_value']),
        float(request.form['commercial_assets_value']),
        float(request.form['luxury_assets_value']),
        float(request.form['bank_asset_value'])
    ]
    
    #Convert the form data into a DataFrame
    new_data = pd.DataFrame([features])
    
    #Preprocess the new data (scaling)
    new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)
    
    #Make predictions
    prediction = model.predict(new_data_scaled)[0]
    
    # Return the result
    if prediction == 1:
        status = 'Approved'
    else:
        status = 'Rejected'
    
    # Render the prediction result on the same page or redirect to a result page
    return render_template('result.html', prediction=status)

if __name__ == '__main__':
    # Ensure the template folder is correctly referenced
    app.run(host='0.0.0.0',port=5000)



