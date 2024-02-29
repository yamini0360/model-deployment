from flask import Flask, render_template, request, redirect
from flask import Flask, render_template, request, redirect
import pickle
import numpy as np

app = Flask(__name__, template_folder='template', static_url_path='/static')

# Load your pre-trained ML model
with open('C:\\model deployment\\template\\stacking_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("request")
        print(type(request)
        # Process form data and perform prediction
        age = float(request.form['age'])
        sex = request.form['sex']  # Keep it as a string
        cp = int(request.form['cp'])
        bp = float(request.form['bp'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Encode 'sex' as numerical value
        sex_encoded = 0 if sex == 'male' else 1
        print("hello")
        input_data = np.array([[age, sex_encoded, cp, bp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        print("worked till here")
        prediction = model.predict(input_data)
        print("prediction successful")
        # Redirect based on final prediction
        if prediction[0] == 1:
            return redirect('/positive')
        else:
            return redirect('/negative')

    except Exception as e:
        # Handle exceptions, such as invalid input data
        return render_template('error.html', error_message=str(e))

# Route for the positive result page
@app.route('/positive')
def positive():
    return render_template('positive.html')

# Route for the negative result page
@app.route('/negative')
def negative():
    return render_template('negative.html')

if __name__ == '__main__':
    app.run(debug=True)








