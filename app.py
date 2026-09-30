from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

# Diabetes Model
diabetes_model = joblib.load(
    "diabetes_model/model/diabetes_model.pkl"
)

# Heart Model
heart_model = joblib.load(
    "heart_model/model/heart_model.pkl"
)
# Liver Model
liver_model = joblib.load(
    "liver_model/model/liver_model.pkl"
)

#Home Pages
@app.route("/")
def home():
    return render_template("index.html")


# DIABETES PREDICTION
@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"])
            ]
            # Prediction
            result = diabetes_model.predict([features])[0]
            if result == 1:
                prediction = "High Diabetes Risk"
            else:
                prediction = "Low Diabetes Risk"
        except Exception as e:
            error = str(e)
    return render_template(
        "diabetes.html",
        prediction=prediction,
        error=error
    )



# Liver Disease Prediction
@app.route("/liver", methods=["GET", "POST"])
def liver():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["age"]),
                float(request.form["total_bilirubin"]),
                float(request.form["direct_bilirubin"]),
                float(request.form["alkphos"]),
                float(request.form["sgpt"]),
                float(request.form["sgot"]),
                float(request.form["total_proteins"]),
                float(request.form["albumin"]),
                float(request.form["ag_ratio"])
            ]

            print("\n========== LIVER PREDICTION ==========")
            print("Features:", features)
            print("Number of features:", len(features))

            # Prediction
            result = liver_model.predict([features])[0]

            print("Model result:", result)

            if result == 1:
                prediction = "High Liver Disease Risk"
            else:
                prediction = "Low Liver Disease Risk"

            print("Final prediction:", prediction)

        except Exception as e:
            print("\n========== LIVER ERROR ==========")
            print(type(e).__name__)
            print(str(e))

            error = str(e)

    return render_template(
        "liver.html",
        prediction=prediction,
        error=error
    )

# Heart Disease Prediction
@app.route("/heart", methods=["GET", "POST"])
def heart():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            features = [
                float(request.form["id"]),
                float(request.form["age"]),
                float(request.form["gender"]),
                float(request.form["height"]),
                float(request.form["weight"]),
                float(request.form["ap_hi"]),
                float(request.form["ap_lo"]),
                float(request.form["cholesterol"]),
                float(request.form["gluc"]),
                float(request.form["smoke"]),
                float(request.form["alco"]),
                float(request.form["active"])
            ]

            print("\n========== HEART PREDICTION ==========")
            print("Features:", features)
            print("Number of features:", len(features))

            result = heart_model.predict([features])[0]

            print("Model result:", result)

            if result == 1:
                prediction = "High Cardiovascular Disease Risk"
            else:
                prediction = "Low Cardiovascular Disease Risk"

            print("Final prediction:", prediction)

        except Exception as e:

            print("\n========== HEART ERROR ==========")
            print(type(e).__name__)
            print(str(e))

            error = str(e)

    return render_template(
        "heart.html",
        prediction=prediction,
        error=error
    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)