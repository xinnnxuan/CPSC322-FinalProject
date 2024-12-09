import pickle

from flask import Flask, request, jsonify, redirect
from flask import render_template
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn import myutils

app = Flask(__name__)

def load_classifier():
    # unpickle header and tree in tree.p
    infile = open('tree.p', 'rb')
    header, tree, y_train = pickle.load(infile)
    decision_tree_classifier = MyDecisionTreeClassifier()
    decision_tree_classifier.header = header
    decision_tree_classifier.tree = tree
    decision_tree_classifier.y_train = y_train
    infile.close()

    return decision_tree_classifier

@app.route('/', methods = ['GET', 'POST'])
def index():
    # prediction = ""
    # if request.method == 'POST':
    #     price = request.form['Price']
    #     year = request.form['Year']
    #     num_ratings = request.form['NumberOfRatings']
    #     print('price from index', price)
    #     print('year from index', year)
    #     print('num ratings from index', num_ratings)
    #     print(request.form)
    #     decision_tree_classifier = load_classifier()
    #     print('tree:', decision_tree_classifier.tree)
    #     prediction = predict_rating([price, year, num_ratings], decision_tree_classifier)
    #     print('prediction:', prediction)
        # render the form on the index page
    return render_template('index.html', prediction="")

@app.route('/predict', methods=['POST'])
def predict():
    # parse the unseen instance values from the query string (from the POST request, they are in the request object
    price = float(request.form.get('Price')) # defaults to None
    year = int(request.form.get('Year'))
    num_ratings = int(request.form.get('NumberOfRatings'))
    print('price from predict', price)
    print('year from predict', year)
    print('num ratings from predict', num_ratings)

    # load the classifier and make a prediction
    decision_tree_classifier = load_classifier()
    print('loaded header from predict', decision_tree_classifier.header)
    print('loaded tree from predict', decision_tree_classifier.tree)
    instance = [price, year, num_ratings]
    prediction = predict_rating(instance, decision_tree_classifier)
    prediction = myutils.most_frequent(prediction)
    if prediction is not None: 
        return render_template('prediction.html', prediction=prediction)
    return 'Error making a prediction', 400

def predict_rating(unseen_instance, classifier):
    try:
        # discretize unseen instance values before prediction
        unseen_instance[0] = price_discretizer(unseen_instance[0])
        unseen_instance[1] = year_discretizer(unseen_instance[1])
        unseen_instance[2] = num_ratings_discretizer(unseen_instance[2])
        print('classifier tree from predict rating', classifier.tree)
        return classifier.predict(unseen_instance)
    except:
        return None

def price_discretizer(price):
    """Discretizes price into 5 categories"""
    if price <= 10:
        return "cheap"
    if price > 10 and price <= 25:
        return "affordable"
    if price > 25 and price <= 50:
        return "average"
    if price > 50 and price <= 1707:
        return "expensive"
    if price > 1707:
        return "very expensive"

def year_discretizer(year):
    """Discretizes year into 5 categories"""
    if year <= 2000:
        return "before 2000"
    if year > 2000 and year <= 2010:
        return "2000-2010"
    if year > 2010:
        return "2010-present"
    
def num_ratings_discretizer(num_ratings):
    """Discretizes number of ratings into 3 categories"""
    if num_ratings <= 500:
        return "few ratings"
    if num_ratings > 500 and num_ratings <= 39273:
        return "some ratings"
    if num_ratings > 39273:
        return "many ratings"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 
    # TODO when deploy app to "production", set debut=False