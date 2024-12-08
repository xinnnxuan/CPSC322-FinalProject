import pickle

from flask import Flask, request, jsonify, redirect
from flask import render_template
from mysklearn.myclassifiers import MyDecisionTreeClassifier

app = Flask(__name__)

def load_classifier():
    # unpickle header and tree in tree.p
    infile = open('tree.p', 'rb')
    header, tree = pickle.load(infile)
    decision_tree_classifier = MyDecisionTreeClassifier()
    decision_tree_classifier.header = header
    decision_tree_classifier.tree = tree
    print('loaded header:', decision_tree_classifier.header)
    print('loaded tree:', decision_tree_classifier.tree)
    infile.close()

    return decision_tree_classifier

@app.route('/', methods = ['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        price = request.form['Price']
        year = request.form['Year']
        num_ratings = request.form['NumberOfRatings']
        print(request.form)
        decision_tree_classifier = load_classifier()
        print('tree:', decision_tree_classifier.tree)
        prediction = predict_rating([price, year, num_ratings], decision_tree_classifier)
        print('prediction:', prediction)
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['GET'])
def predict():
    # parse the unseen instance values from the query string
    # they are in the request object
    price = request.args.get('Price') # defaults to None
    year = request.args.get('Year')
    num_ratings = request.args.get('NumberOfRatings')
    instance = [price, year, num_ratings]
    decision_tree_classifier = load_classifier()
    prediction = predict_rating(instance, decision_tree_classifier)
    if prediction is not None: 
        return jsonify({'prediction': prediction}), 200
    return 'Error making a prediction', 400

def predict_rating(unseen_instance, classifier):
    try:
        return classifier.predict(unseen_instance)
    except:
        return None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 
    # TODO when deploy app to "production", set debut=False