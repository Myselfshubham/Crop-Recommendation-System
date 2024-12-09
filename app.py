from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('StandardScaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session handling

# Crop-to-image mapping
crop_image_dict = {
    1: "rice.jpg", 2: "maize.jpg", 3: "jute.jpg", 4: "cotton.jpg", 5: "coconut.jpg",
    6: "papaya.jpg", 7: "orange.jpg", 8: "apple.jpg", 9: "muskmelon.jpg", 10: "watermelon.jpg",
    11: "grapes.jpg", 12: "mango.jpg", 13: "banana.jpg", 14: "pomegranate.jpg", 15: "lentil.jpg",
    16: "blackgram.jpg", 17: "mungbean.jpg", 18: "mothbeans.jpg", 19: "pigeonpeas.jpg",
    20: "kidneybeans.jpg", 21: "chickpea.jpg", 22: "coffee.jpg"
}

# Home route


@app.route("/", methods=["GET"])
def index():
    # Retrieve results from the session, if available
    result = session.get("result")
    crop_image = session.get("crop_image")
    # Clear the session after rendering
    session.clear()
    return render_template("index.html", result=result, crop_image=crop_image)

# Predict route


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Prepare features for prediction
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Process features with the model
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        # Crop name mapping
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
                     7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
                     12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
                     17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
                     21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            crop_image = crop_image_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            crop_image = "default.jpg"

        # Store results in session and redirect to the index page
        session["result"] = result
        session["crop_image"] = crop_image
        return redirect(url_for('index'))

    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('index'))



#uncomment this for runnin in local environment
#if __name__ == "__main__":
    #app.run(debug=True)
