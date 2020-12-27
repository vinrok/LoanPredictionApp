from flask import Flask,request, url_for, redirect, render_template, jsonify, flash, Markup
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import os

# model
import joblib


pkl_filename = 'static/models/loan_predn_pipeline_lr_clf.pkl'

# load model
model = joblib.load(pkl_filename)
#print('Model loaded: ', model)

cols = ['Gender','Married','Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']


app = Flask(__name__,
			static_url_path='',
            static_folder='static',
            template_folder='templates')

CORS(app, support_credentials=True)
app.config['SECRET_KEY'] =  os.urandom(24)


@app.route('/')
@app.route('/index')
@cross_origin(supports_credentials=True)
def index():
    return render_template("index.html")

@app.route('/apply')
@cross_origin(supports_credentials=True)
def apply():
    return render_template("predictions.html")

@app.route('/predict',methods=['POST'])
@cross_origin(supports_credentials=True)
def predict():
    int_features = [int(x) if x.isnumeric() else x for x in request.form.values()]
    final = np.array(int_features)
    print('input records ', final)
    data_unseen = pd.DataFrame([final], columns = cols)

    data_unseen['Credit_History'] = data_unseen['Credit_History'].astype('float')
    data_unseen['Dependents'] = data_unseen['Dependents'].astype(str)

    print(data_unseen)

    # predict
    pred_prob = model.predict_proba(data_unseen)[0, 1]
    print('prob: ', pred_prob)

    if pred_prob > 0.5:
        conclusion = 'Loan APPROVED!'
        category = 'success'
    else:
        conclusion = 'Loan REJECTED!'
        category = 'danger'

    message = Markup(
            f'''
            <b>Probability of loan approval</b>: {pred_prob*100:.2f}% <br>
            <b>Offer</b>: {conclusion}
            '''
        )

    flash(message, category=category)


    return render_template('predictions.html', title='Using ML!')

@app.route('/predict_loan_api',methods=['POST'])
@cross_origin(supports_credentials=True)
def predict_loan_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(data=data_unseen)
    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
