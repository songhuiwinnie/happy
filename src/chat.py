# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from src.response import get_response, find_dates, find_places
from src.initialize import initialize
from src.response import COUNTRIES
import flask_sijax
import os


path = os.path.join('.', os.path.dirname(__file__), 'static/js/sijax/')
app = Flask(__name__)
app.config['SIJAX_STATIC_PATH'] = path
app.config['SIJAX_JSON_URI'] = '../static/js/sijax/json2.js'
flask_sijax.Sijax(app)


# Chatting api
@app.route("/api/query")
def query():
    message = str(request.args.get('message'))
    message_response = str(get_response(message, seq2seq_models, wv_model))
    date_departure, date_return = find_dates(message)
    place_from, place_to = find_places(message)
    response = dict(message=message_response, date_departure=date_departure, date_return=date_return,
                    place_from=place_from, place_to=place_to)
    return jsonify(response)


@app.route("/api/countries")
def countries():
    response = dict(countries=COUNTRIES)
    return jsonify(response)


# Load chatting page
@app.route('/')
def index():
    return render_template('chat.html')


if __name__ == "__main__":
    seq2seq_models, wv_model = initialize()
    app.run(debug=False, port=8888)
