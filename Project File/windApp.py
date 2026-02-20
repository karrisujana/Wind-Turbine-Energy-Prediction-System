import numpy as np
from flask import Flask, request, render_template
import joblib
import requests

app = Flask(__name__)

# Load the saved model
model = joblib.load('power_prediction.sav')

@app.route('/')
def home():
    # Renders the landing page
    return render_template('intro.html')

@app.route('/predict')
def predict_page():
    # Renders the input form page
    return render_template('predict.html')

@app.route('/windapi', methods=['POST'])
def windapi():
    # Gets current weather from OpenWeather API
    city = request.form.get('city')
    apikey = "43ce69715e2133b2300e0f8f7289befd" # Use your own key if this one is inactive
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}&units=metric"
    
    resp = requests.get(url).json()
    
    # Extract data to display on UI
    temp = f"{resp['main']['temp']} °C"
    humid = f"{resp['main']['humidity']} %"
    pressure = f"{resp['main']['pressure']} hPa"
    speed = f"{resp['wind']['speed']} m/s"
    
    return render_template('predict.html', temp=temp, humid=humid, pressure=pressure, speed=speed)

@app.route('/y_predict', methods=['POST'])
def y_predict():
    # Grabbing by name 's', 't', and 'p' from the HTML form
    s = float(request.form.get('s'))
    t = float(request.form.get('t'))
    p = float(request.form.get('p'))
    
    features = [np.array([s, t, p])]
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('predict.html', 
                           prediction_text=f'Predicted Energy: {output} kW')
if __name__ == "__main__":
    app.run(debug=True)                 