from flask import Flask, render_template, request
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import jsonify
from numpy import argmax

app = Flask(__name__)


dic = {0: 'Covid 19', 1: 'Normal', 2: 'Viral Pneumonia'}

model = load_model('sequential.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(256, 256, 3))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    i = np.vstack([i])
    p = model.predict(i)
    print(p)
    r = argmax(p)
    print(r)
    return dic[r]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/test-now", methods=['GET', 'POST'])
def upload():
    return render_template("test_now.html")


@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("test_now.html", prediction=p, img_path=img_path)
    # return jsonify(p)

if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
