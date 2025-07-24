from flask import Flask, render_template, request
import pickle
import numpy as np
import os
app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'car.pkl')
model = pickle.load(open(model_path, 'rb'))


brand_map = {
    "Maruti Suzuki": 0, "Hyundai": 1, "Honda": 2, "Toyota": 3,
    "Mahindra": 4, "Mercedes-Benz": 5, "Tata": 6, "BMW": 7,
    "Volkswagen": 8, "Audi": 9, "Ford": 10, "Renault": 11,
    "Skoda": 12, "Kia": 13, "Chevrolet": 14, "MG": 15,
    "Nissan": 16, "Jeep": 17, "Land Rover": 18, "Volvo": 19,
    "Mini": 20, "Jaguar": 21, "Porsche": 22, "Fiat": 23,
    "Datsun": 24, "Lexus": 25, "Mitsubishi": 26, "Isuzu": 27,
    "Force": 28, "Ambassador": 29, "Rolls-Royce": 30, "Bajaj": 31,
    "Opel": 32, "Ssangyong": 33, "Aston Martin": 34, "ICML": 35,
    "Ashok": 36, "Bentley": 37, "Maserati": 38
}

@app.route('/')
def home():
    return render_template('index.html', brands=list(brand_map.keys()), prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form['brand']
    year = int(request.form['year'])
    kms = int(request.form['kms'])
    vehicle_type = 0 if request.form['vehicle_type'] == 'Manual' else 1
    owner = 0 if request.form['owner'] == 'first' else 1
    fuel_type = 0 if request.form['fuel_type'] == 'Petrol' else 1 if request.form['fuel_type'] == 'Diesel' else 2
    brand_encoded = brand_map[brand]
    
    X_test = np.array([[brand_encoded, year, kms, vehicle_type, owner, fuel_type]])
    yp = model.predict(X_test)[0]
    
    return render_template('index.html', brands=list(brand_map.keys()), prediction=round(yp, 2))

if __name__ == '__main__':
    app.run(debug=True)
