from flask import Flask, render_template, request
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
app = Flask(__name__)
model = load_model("dl_model.h5")
scaler = pickle.load(open("dl_model.pkl","rb"))
@app.route('/')
def home():
    month = ["Jan","Feb","Mar","Apr","May","Jun"]
    sales = [100,250,350,200,280,300]

    plt.plot(month,sales)
    plt.title("Monthly Sale")
    plt.xlabel("Months")
    plt.ylabel("Sales")
    # plt.show()

    image = io.BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)

    image_64bit = base64.b64encode(image.getvalue()).decode('utf-8')

    return render_template('index.html', chart=image_64bit)

@app.route('/submit', methods=['POST'])
def submit():
    tenure = request.form['num1']
    MonthlyCharges = request.form['num2']
    

    # tenure = input("Enter tenure in months: ")
    # MonthlyCharges = input("Enter Monthly Charges: ")

    customer = pd.DataFrame({
        'tenure' : [tenure],
        'MonthlyCharges' : [MonthlyCharges]
    })

    customer = scaler.transform(customer)

    result = model.predict(customer)
    print("Customer will Stay Probability: ", round(result[0][0] * 100,2), "%")
    print("Customer wwill Churn Probability: ",round(1-result[0][0] * 100,2), "%")

    if result[0][0] > 0.5:
        # return f"Customer will Churn"
        return render_template('result.html', result="Customer will Churn")
        print("Customer will Churn")
    else:
        # return f"Customer will Stay"
        return render_template('result.html', result="Customer will Stay")
        print("Customer will Stay")



if __name__ == '__main__':
    app.run(debug=True)


    