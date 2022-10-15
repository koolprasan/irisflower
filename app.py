import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('FlowerPredictor.mdl', 'rb'))
#ohe = pickle.load(open('StateEncoder.obj','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    labels = {0: 'iris-setosa',
              1: 'iris-versicolor',
              2: 'iris-virginica'}
    int_features = [float(x) for x in request.form.values()]

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = np.vectorize(labels.__getitem__)(prediction)

    return render_template('index.html', prediction_text='The Flower is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)