from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
train_data = pd.read_csv('train.csv')
X = train_data.drop(['TARGET(PRICE_IN_LACS)'], axis=1)
y = train_data['TARGET(PRICE_IN_LACS)']
X_train, _, _, _ = train_test_split(X, y, test_size=0.1, random_state=42)
n_features = X_train.shape[1]
try:
    loaded_model = joblib.load('E:\house price prediction\gradient_boosting_model (1).pkl')
except FileNotFoundError:
    print("Model file not found. Please check the file path.")
    exit()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
           user_input = {
                'POSTED_BY': 1 if request.form['POSTED_BY'] == 'Owner' else 0,
                'UNDER_CONSTRUCTION': int(request.form['UNDER_CONSTRUCTION']),
                'RERA': int(request.form['RERA']),
                'BHK_NO.': int(request.form['BHK_NO.']),
                'BHK_OR_RK': 1 if request.form['BHK_OR_RK'] == 'BHK' else 0,
                'SQUARE_FT': int(request.form['SQUARE_FT']),
                'READY_TO_MOVE': int(request.form['READY_TO_MOVE']),
                'RESALE': int(request.form['RESALE']),
            }

        except ValueError:
            return render_template('error.html', message='Invalid input. Please enter valid values.')
        try:
            prediction = loaded_model.predict(pd.DataFrame(user_input, index=[0]))[0]
            # Round the prediction to the nearest integer and convert to lacs
            prediction_in_lacs = round(prediction) / 100000
        except Exception as e:
            return render_template('error.html', message='An error occurred while making the prediction. {}'.format(str(e)))
        return render_template('result.html', prediction=prediction_in_lacs)

if __name__ == '__main__':
    app.run(debug=True)
