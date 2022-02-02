import joblib
import numpy as np
from flask import Blueprint, render_template, request, app
from flask_login import login_required, current_user
from . import db #dot is where u actualy are

MODEL_PATH = '/Users/anitachrust/Desktop/pytong/project/model.sav'
main = Blueprint('main', __name__)


@main.route('/')
def index():
    return render_template("index.html")


@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name, surname=current_user.surname)


@main.route('/ml')
@login_required
def ml():
    return render_template("ml.html")


@login_required
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load(MODEL_PATH)
    result = loaded_model.predict(to_predict)
    return result[0]


@main.route('/ml', methods=['POST', 'GET'])
@login_required
def ml_post():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = round(float(ValuePredictor(to_predict_list)), 2)
        return render_template("ml.html", result=result)
