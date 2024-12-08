import pickle

from flask import Flask, request, jsonify
from mysklearn.myclassifiers import MyDecisionTreeClassifier

app = Flask(__name__)

def load_model():
    # unpickle header and tree in tree.p
    infile = open('tree.p', 'rb')
    header, tree = pickle.load(infile)
    infile.close()
    return header, tree

@app.route('/')
def index():
    # return content and status code
    return "<h1>Welcome to the wine predictor app</h1>", 200

@app.route('/predict')
def predict():
    # parse the unseen instance values from the query string
    # they are in the request object
    price = request.args.get('Price') # defaults to None
    year = request.args.get('Year')
    num_ratings = request.args.get('NumberOfRatings')
    instance = [price, year, num_ratings]
    header, tree = load_model()
    decision_tree_classifier = MyDecisionTreeClassifier()
    decision_tree_classifier.header = header
    decision_tree_classifier.tree = tree
    prediction = tree.predict(instance)
    if prediction is not None: 
        return jsonify({'prediction': prediction}), 200
    return 'Error making a prediction', 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 
    # TODO when deploy app to "production", set debut=False