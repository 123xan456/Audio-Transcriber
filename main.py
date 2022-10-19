import whisper
import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
model = whisper.load_model("small")
df = pd.read_csv("result.csv")


@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/result", methods=["POST"])
def transcribe():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    result = model.transcribe(str(UPLOAD_FOLDER + "/" + filename), language="en")
    os.remove("uploads/" + filename)
    return render_template("result.html", result=result["text"], filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
