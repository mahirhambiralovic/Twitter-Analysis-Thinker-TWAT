# In order to run with "flask run"
# Write the two following commands in the beginning of the session.
#   $ export FLASK_APP=application.py
#   $ export FLASK_ENV=development

from flask import Flask, flash, redirect, render_template, request, session
from trainer import corpusGenerator, load_obj_files, predict
from helpers import get_gif_url

app = Flask(__name__)
cv, log_model = load_obj_files()

@app.route("/", methods=["GET", "POST"])
def index():
    learning_gif_url = get_gif_url("learning")

    # Get search query
    if request.method == "POST":
        user_input = request.form.get("user_input")
    # If no query, user_input is 'You're awesome!'
    else:
        user_input = "You're awesome!"
    print("user_input = ", user_input)

    # Predict and get sentiment text
    sentiment = predict(user_input, cv, log_model)
    return render_template("index.html", user_input=user_input, sentiment=sentiment, learning_gif_url=learning_gif_url)

@app.route("/about", methods=["GET"])
def about():
    cat_gif_url = get_gif_url("cat")
    return render_template("about.html", cat_gif_url=cat_gif_url)
