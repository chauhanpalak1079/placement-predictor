from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open("placement_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")
@app.route("/result",methods=["POST"])
def result():

    cgpa = float(request.form["cgpa"])
    internships = int(request.form["internships"])
    projects = int(request.form["projects"])
    aptitude = int(request.form["aptitude"])
    softskills = int(request.form["softskills"])

    data = [[cgpa, internships, projects, aptitude, softskills]]

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        message = "High Placement Potential"
        advice = "Keep improving your projects and communication skills to secure top job opportunities."
    else:
        message = "Placement Chances Need Improvement"
        advice = "Focus on increasing your technical skills, internships, and aptitude preparation."

    return render_template(
        "result.html",
        prediction=message,
        probability=round(probability,2),
        advice=advice
    )
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)