"""Programmer: Lindsey Bodenbender
Class: CPSC 322 Fall 2024
11/18/2024

Description: General utility functions"""

import numpy as np
from mysklearn.mypytable import MyPyTable

X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ] 
y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# TODO: in fit(), programmatically build header and attribute_domains
# using X_train. perhaps store as attributes of MyDecisionTreeClassifier
header = ["att0", "att1", "att2", "att3"]
attribute_domains = {"att0": ["Junior", "Mid", "Senior"], 
        "att1": ["Java", "Python", "R"],
        "att2": ["no", "yes"], 
        "att3": ["no", "yes"]}
# how to represent trees in Python?
# 1. nested data structures (like dictionaries, lists, etc.)
# 2. OOP (e.g. MyTree class)
# we will use a nested list approach
# at element 0: data type (Attribute, Value, Leaf)
# at element 1: data value (attribute name, 
# value name, class label)
# rest of elements: depends on the type
# example!
interview_tree_solution =   ["Attribute", "att0", 
                                ["Value", "Junior", 
                                    ["Attribute", "att3",
                                        ["Value", "no",
                                            ["Leaf", "True", 3, 5]
                                        ],
                                        ["Value", "yes",
                                            ["Leaf", "False", 2, 5]
                                        ]
                                    ]
                                ],
                                ["Value", "Mid",
                                    ["Leaf", "True", 4, 14]
                                ],
                                ["Value", "Senior",
                                    ["Attribute", "att2",
                                        ["Value", "no",
                                            ["Leaf", "False", 3, 5]
                                        ],
                                        ["Value", "yes",
                                             ["Leaf", "True", 2, 5]
                                        ]
                                    ]
                                ]
                            ]

def extract_header_att_domains(X_train, y_train, header=None):
    if header is None:
        header = ['att' + str(i) for i in range(len(X_train[0]))]
    attribute_domains = {}
    for i, val in enumerate(header):
        if i != -1:
            col = extract_col(X_train, header, header[i])
            domain = get_unique_labels(col)
            attribute_domains[header[i]] = domain
    
    return header, attribute_domains

def count_class_labels(data, labels):
    counts = [0 for i in range(len(labels))]
    for i in data:
        counts[labels.index(i[-1])] += 1
    return counts

def compute_entropy(posteriors):
    """Helper function to compute the entropy of a list of posteriors"""
    entropy = 0
    if posteriors[0] == 0:
        return entropy
    for posterior in posteriors:
        entropy += -(np.log2(posterior)) * posterior
    return entropy

def most_frequent(col):
    """Returns the value in a list that is the most frequent"""
    unique_vals = get_unique_labels(col)
    counts = [0 for i in range(len(unique_vals))]
    for i in col:
        counts[unique_vals.index(i)] += 1
    max_val = 0
    for i in counts:
        if i > max_val:
            max_val = i
    return unique_vals[counts.index(max_val)]

def print_matrix(matrix):
    """Prints a matrix in a form that is easy to read"""
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            print(matrix[i][j], "   ", end="")
        print("\n", end="")

def display_kfold_results(mean_accuracy, mean_error, k, precision, recall, matrix, f1_score, classifier_name):
    print(f' ========================== \n    Predictive Accuracy \n ========================== \n Random Subsample k={k}, {classifier_name} Classifier: accuracy = {mean_accuracy}, error rate = {mean_error}, precision = {precision}, recall = {recall}, F1 score = {f1_score}')
    print_matrix(matrix)

def create_posterior_dict(class_labels, posteriors, all_categories):
    """Takes a list of posteriors and formats them in a dictionary with the attribute values as keys"""
    posterior_dict = {}

    for class_idx, class_label in enumerate(class_labels):
        # create a dictionary for the current class label
        posterior_dict[class_label] = {}
        
        # to traverse posteriors
        flat_index = 0

        # iterate over attributes and their possible categories
        for attr_idx, categories in enumerate(all_categories):
            for category in categories:
                key = f"att{attr_idx + 1}={category}"
                
                # assign the corresponding value from the posteriors list
                posterior_dict[class_label][key] = posteriors[class_idx][flat_index]
                
                # move to the next flat index in posteriors
                flat_index += 1

    return posterior_dict

def calc_tp_fp(y_true, y_pred, labels, pos_label):
    """Calculates the true positives and false positives from a list of true and predicted y values"""
    tp = 0
    fp = 0

    if labels is None:
        labels = get_unique_labels(y_true)
    if pos_label is None:
        pos_label = labels[0]

    for i, val in enumerate(y_true):
        if y_true[i] == y_pred[i] and val == pos_label:
            tp += 1
        if y_pred[i] == pos_label and y_true[i] != y_pred[i]:
            fp += 1

    return tp, fp

def calc_tp_fn(y_true, y_pred, labels, pos_label):
    """Calculates the true positives and false negatives from a list of true and predicted y values"""

    tp = 0
    fn = 0

    if labels is None:
        labels = get_unique_labels(y_true)
    if pos_label is None:
        pos_label = labels[0]

    for i, val in enumerate(y_true):
        if y_true[i] == y_pred[i] and val == pos_label:
            tp += 1 
        if y_pred[i] != pos_label and y_true[i] == pos_label:
            fn += 1
    return tp, fn

def predict_labels(products, class_labels):
    """Returns label predictions for slices of the products array"""
    predictions = []
    for i in range(0, len(products), len(class_labels)):
        product_slice = products[i:i + len(class_labels)]
        max_index = product_slice.index(max(product_slice))
        predictions.append(class_labels[max_index])
    return predictions

def extract_col(data, col_labels, col_label):
    """Extract a column with the given label"""
    col = []
    if not col_labels is None:
        if isinstance(col_label, str):
            key = col_labels.index(col_label)
        else:
            key = col_label
        for row in data:
            for i, val in enumerate(row):
                col.append(row[key])
        return col
    else:
        key = col_label
        for row in data:
            for i, val in enumerate(row):
                col.append(row[key])
        return col

def get_unique_labels(data):
    """Returns all unique items in a list"""
    labels = []
    for i in data:
            if i not in labels:
                labels.append(i)
    return labels

def display_num_unique_labels(col, name):
    """Displays the number of unique labels for a given column"""
    unique_vals = get_unique_labels(col)
    num_unique = len(col)
    print(f'There are {num_unique} unique {name}')

def count_nv(data):
    count = 0
    for row in data:
        for i in row:
            if i == "N.V.":
                count += 1
    return count

def rating_discretizer(rating):
    """Discretizes rating into 5 categories"""
    if rating <= 3.2:
        return "poor"
    if rating > 3.2 and rating <= 3.7:
        return "ok"
    if rating > 3.7 and rating <= 4.0:
        return "average"
    if rating > 4.0 and rating <= 4.2:
        return "good"
    if rating > 4.2 and rating <= 4.9:
        return "excellent"

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
    if price > 1707 and price <= 3411:
        return "very expensive"

def year_discretizer(year):
    """Discretizes year into 5 categories"""
    if year <= 2000:
        return "before 2000"
    if year > 2000 and year <= 2010:
        return "2000-2010"
    if year > 2010 and year <= 2020:
        return "2010-2020"
    
def num_ratings_discretizer(num_ratings):
    """Discretizes number of ratings into 3 categories"""
    if num_ratings <= 500:
        return "few ratings"
    if num_ratings > 500 and num_ratings <= 39273:
        return "some ratings"
    if num_ratings > 39273 and num_ratings <= 94257:
        return "many ratings"

def discretizer(y_predicted):
    """Discretizes into two categories, high or low"""
    if y_predicted >= 100:
            return "high"
    else:
        return "low"

def mpg_discretizer(y_predicted):
    """Discretizes into DOE mpg ratings"""
    if y_predicted <= 13:
        return 1
    if y_predicted > 13 and y_predicted <= 15:
        return 2
    if y_predicted >= 15 and y_predicted <= 16:
        return 3
    if y_predicted >= 16 and y_predicted <= 20:
        return 4
    if y_predicted >= 20 and y_predicted <= 24:
        return 5
    if y_predicted >= 24 and y_predicted <= 27:
        return 6
    if y_predicted >= 27 and y_predicted <= 31:
        return 7
    if y_predicted >= 31 and y_predicted <= 37:
        return 8
    if y_predicted >= 37 and y_predicted <= 45:
        return 9
    if y_predicted >= 45:
        return 10

def mpg_discretizer_list(y_predicted):
    """Discretizes a list of predicted values into DOE mpg ratings"""
    ratings = []
    for i in y_predicted:
        if i <= 13:
            ratings.append(1)
        if i > 13 and i < 15:
            ratings.append(2)
        if i >= 15 and i <= 16:
            ratings.append(3)
        if i >= 16 and i <= 20:
            ratings.append(4)
        if i >= 20 and i <= 24:
            ratings.append(5)
        if i >= 24 and i <= 27:
            ratings.append(6)
        if i >= 27 and i <= 31:
            ratings.append(7)
        if i >= 31 and i <= 37:
            ratings.append(8)
        if i >= 37 and i <= 45:
            ratings.append(9)
        if i >= 45:
            ratings.append(10)
    return ratings

def display_results(X_test, table, pred, actual, accuracy):
    """Displays the test instances with their predicted and actual values and accuracy"""
    for i in range(len(X_test)):
        print('Instance:', table.data[i])
        print(f'Class: {pred[i]} Actual: {actual[i]}')
    print('Accuracy:', accuracy)

def calculate_accuracy(predicted, actual):
    correct_count = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            correct_count += 1
    accuracy = correct_count / len(predicted)
    return accuracy

def compute_euclidean_distance(X_train, X_test):
    """Calcualtes the euclidean distance between the points in two vectors"""
    distances = []
    for x in X_test:
        # calculate the distance between each item in X_train and X_test
        dist = np.sqrt(sum([(X_train[i] - x[i]) ** 2 for i in range(len(X_train))]))
        distances.append(dist)
    return distances

def compute_categorical_distance(X_train, X_test):
    """Computes the distance between all the categorical attributes in X_train and X_test
    A distance of 1 is assigned if the attributes are the same and 0 if they are different"""
    distances = []
    for i, x in enumerate(X_test):
        for j, val in enumerate(X_train):
            if X_test[i] == X_train[j]:
                dist = 1
            else:
                dist = 0
            distances.append(dist)
    return distances

def choose_rand_indexes(data, num_indexes, rand_seed=0):
    """Chooses random indexes"""
    np.random.seed(rand_seed)
    rand_indexes = [np.random.randint(0, len(data)) for i in range(num_indexes)]
    return rand_indexes

def extract_test_set(data, test_indices):
    """Creates a list of data from a table from the specified indices"""
    X_test = []
    for i, row in enumerate(data):
        if i in test_indices:
            X_test.append(row)
    return X_test

def extract_train_set(data, test_indices):
    """Creates a list of data from a table from the specified indices"""
    train_set = []
    for i, row in enumerate(data):
        if i not in test_indices:
            train_set.append(row)
    return train_set

def convert_to_2D(list_to_convert):
    """Converts a list to 2D"""
    # if the list is a 2D list
    if isinstance(list_to_convert[0], list):
        return list_to_convert
    else:
        converted_list = []
        for i in list_to_convert:
            converted_list.append([i])
        return converted_list
    
def convert_to_2D_specify_num_items(list_to_convert, num_items):
    """Converts a list to 2D if it isn't already"""
    # if the list is a 2D list
    if isinstance(list_to_convert[0], list):
        return list_to_convert
    else:
        converted_list = []
        # for i, val in enumerate(list_to_convert):
        test = len(list_to_convert) / num_items
        for i in range(int(len(list_to_convert) / num_items)):
            row = []
            for j in range(num_items):
                row.append(list_to_convert[i])
            converted_list.append(row)
        return converted_list

def convert_to_1D(list_to_convert):
    """Converts a list to 1D format"""
    converted_list = []
    # if the list is a 2D list
    if isinstance(list_to_convert[0], list):
        for i, row in enumerate(list_to_convert):
            for j, col in enumerate(row):
                converted_list.append(list_to_convert[i][j])
        return converted_list
    else:
        return list_to_convert

def normalize(col, col_min, col_max):
    """Normalizes a column based on max and min values"""
    col_min = min(col)
    col_max = max(col)
    col_range = col_max - col_min

    col_normalized = [((val - col_min) / col_range) for val in col]
    return col_normalized 

def combine_cols(col1, col2, col3):
    """Returns a list containing the contents of three columns"""
    cols_combined = []
    for i in col1:
        cols_combined.append(i)
    for i in col2:
        cols_combined.append(i)
    for i in col3:
        cols_combined.append(i)
    return cols_combined

def reformat_cols(cols_combined, col1, col2, col3):
    """Reformats columns to be in 2D list with each element containing parallel column elements"""
    cols_reformatted = []
    # col1, 2 and 3 should be parallel
    for _ in range(int(len(col1) / 3)):
        for i in range(3):
            row = []
            row.append(col1[i])
            row.append(col2[i])
            row.append(col3[i])
            cols_reformatted.append(row)
    return cols_reformatted

def reformat_cols_normalized(cols_normalized):
    """Formats three columns into a 2D list where each element is an element in a column parallel to the other columns"""
    cols_reformatted = []
    for i in range(int(len(cols_normalized) / 3)):
        row = []
        row.append(cols_normalized[i])
        row.append(cols_normalized[i + 1])
        row.append(cols_normalized[i + 2])
        i += 2
        cols_reformatted.append(row)
    return cols_reformatted

def randomize_in_place(alist, parallel_list=None):
    """Randomize up to two lists in place, if two lists are given keeps the parallel order of them"""
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def separate_data(X, test_index):
    """Separates data into training and testing sets"""
    train_data = []
    test_data = []
    if isinstance(X[0], list):
        pass
    for i, val in enumerate(X):
        if i != test_index:
            train_data.append(X[i])
        else:
            test_data.append(X[i])
    
    return train_data, test_data

def separate_data_convert_to_1D(X, test_index):
    """Separates data into testing and training sets and converts the format to 1D"""
    train_data = []
    test_data = []
    if isinstance(X[0], list):
        X = convert_to_1D(X)
        pass
    for i, val in enumerate(X):
        if i != test_index:
            train_data.append(X[i])
        else:
            test_data.append(X[i])
    
    return train_data, test_data

def pretty_print_matrix(matrix, labels):
    """Prints a matrix with the MPG ranking title"""
    for i, label in enumerate(labels):
        if i == 0:
            print("MPG Ranking  ", end="")
        print(label, "  ", end="")
    print("\n===============================================", end="")
    for i, row in enumerate(matrix):
        print('\n', end="")
        for j, val in enumerate(row):
            # check it's not the totals column
            if j == 0 and i < len(matrix) - 1:
                print("      ", labels[i], "   ", val, "  ", end="")
            else:
                print(val, "  ", end="")

def form_X_Y_train():
    """Forms X_train and y_train from the titanic dataset"""
    filename = "input_data/titanic.csv"
    mpt = MyPyTable()
    mpt.load_from_file(filename)

    # extract X_train columns
    class_col = mpt.get_column('class')
    age = mpt.get_column('age')
    sex = mpt.get_column('sex')

    # combine columns to form X_train
    X_train = combine_cols(class_col, age, sex)
    # reformat so parallel to y_train
    X_train = reformat_cols_normalized(X_train)

    y_train = mpt.get_column('survived')

    return X_train, y_train

def form_X_Y_train_data():
    """Forms X_train and y_train from the data dataset"""

    filename = "input_data/data.csv"
    mpt = MyPyTable()
    mpt.load_from_file(filename)

    # extract X_train columns
    class_col = mpt.get_column('class')
    age = mpt.get_column('age')
    sex = mpt.get_column('sex')

    # combine columns to form X_train
    X_train = combine_cols(class_col, age, sex)
    # reformat so parallel to y_train
    X_train = reformat_cols_normalized(X_train)

    y_train = mpt.get_column('survived')

    return X_train, y_train