"""Programmer: Lindsey Bodenbender
Class: CPSC 322 Fall 2024
Programming Assignment Final project
12/4/2024

Description: Tests for classifiers"""

import numpy as np
# pylint: skip-file
from scipy import stats
from scipy.stats import linregress
from mysklearn import myutils

from mysklearn.mypytable import MyPyTable

from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

from mysklearn.myclassifiers import MyNaiveBayesClassifier

from mysklearn.myclassifiers import MyDecisionTreeClassifier

def test_decision_tree_classifier_fit():
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
    decision_tree_classifier = MyDecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    assert decision_tree_classifier.tree == interview_tree_solution

    iphone_tree_solution = ['Attribute', 'att0', 
                                ['Value', 1, 
                                    ['Attribute', 'att1', 
                                        ['Value', 1, 
                                            ['Leaf', 'yes', 1, 5]], 
                                        ['Value', 2, 
                                            ['Attribute', 'att2', 
                                                ['Value', 'excellent', 
                                                    ['Leaf', 'yes', 1, 2]], 
                                                ['Value', 'fair', 
                                                    ['Leaf', 'no', 1, 2]
                                                ]
                                            ]
                                        ], 
                                        ['Value', 3, 
                                            ['Leaf', 'no', 2, 5]
                                        ]
                                    ]
                                ],
                                ['Value', 2, 
                                    ['Leaf', 'yes', 8, 15]
                                ]
                            ]
    X_train = [
            [1, 3, "fair"],
            [1, 3, "excellent"],
            [2, 3, "fair"],
            [2, 2, "fair"],
            [2, 1, "fair"],
            [2, 1, "excellent"],
            [2, 1, "excellent"],
            [1, 2, "fair"],
            [1, 1, "fair"],
            [2, 2, "fair"],
            [1, 2, "excellent"],
            [2, 2, "excellent"],
            [2, 3, "fair"],
            [2, 2, "excellent"],
            [2, 3, "fair"]  
            ]
    y_train = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    decision_tree_classifier.fit(X_train, y_train)
    assert decision_tree_classifier.tree == iphone_tree_solution

def test_decision_tree_classifier_predict():
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
    X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]

    decision_tree_classifier = MyDecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    y_predicted = decision_tree_classifier.predict(X_test)

    assert y_predicted == ['True', 'False']

    X_train = [
            [1, 3, "fair"],
            [1, 3, "excellent"],
            [2, 3, "fair"],
            [2, 2, "fair"],
            [2, 1, "fair"],
            [2, 1, "excellent"],
            [2, 1, "excellent"],
            [1, 2, "fair"],
            [1, 1, "fair"],
            [2, 2, "fair"],
            [1, 2, "excellent"],
            [2, 2, "excellent"],
            [2, 3, "fair"],
            [2, 2, "excellent"],
            [2, 3, "fair"]  
            ]
    y_train = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]

    decision_tree_classifier.fit(X_train, y_train)
    y_predicted = decision_tree_classifier.predict(X_test)
    assert y_predicted == ['yes', 'yes']

def test_naive_bayes_classifier_fit():
    classifier = MyNaiveBayesClassifier()

    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    classifier.fit(X_train_inclass_example, y_train_inclass_example)

    assert np.allclose(classifier.priors['yes'], 0.625) and np.allclose(classifier.priors['no'], 0.375)
    assert classifier.posteriors['yes']['att1=1'] == 0.8 and classifier.posteriors['yes']['att1=2'] == 0.2 and classifier.posteriors['yes']['att2=5'] == 0.4 and classifier.posteriors['yes']['att2=6'] == 0.6
    assert classifier.posteriors['no']['att1=1'] == 2/3 and classifier.posteriors['no']['att1=2'] == 1/3 and classifier.posteriors['no']['att2=5'] == 2/3 and classifier.posteriors['no']['att2=6'] == 1/3
    
    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]

    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    classifier.fit(X_train_iphone, y_train_iphone)
    
    assert np.allclose(classifier.priors['no'], 0.33333333) and np.allclose(classifier.priors['yes'], 0.6666666)
    assert classifier.posteriors['no']['att1=1'] == 0.6 and classifier.posteriors['no']['att1=2'] == 0.4 and classifier.posteriors['no']['att2=3'] == 0.4 and classifier.posteriors['no']['att2=2'] == 0.4 and classifier.posteriors['no']['att2=1'] == 0.2 and classifier.posteriors['no']['att3=fair'] == 0.4 and classifier.posteriors['no']['att3=excellent'] == 0.6
    assert classifier.posteriors['yes']['att1=1'] == 0.2 and classifier.posteriors['yes']['att1=2'] == 0.8 and classifier.posteriors['yes']['att2=3'] == 0.3 and classifier.posteriors['yes']['att2=2'] == 0.4 and classifier.posteriors['yes']['att2=1'] == 0.3 and classifier.posteriors['yes']['att3=fair'] == 0.7 and classifier.posteriors['yes']['att3=excellent'] == 0.3

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test = [["weekday", "winter", "high", "heavy"]]
    expected_values = [0.6428571428571429, 0.14285714285714285, 0.14285714285714285, 0.07142857142857142, 0.2857142857142857, 0.14285714285714285, 0.42857142857142855, 0.14285714285714285, 0.35714285714285715, 0.2857142857142857, 0.35714285714285715, 0.35714285714285715, 0.5714285714285714, 0.07142857142857142, 0.5, 0.5, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0.5, 1, 0, 0, 0, 0, 0.666666, 0, 0.3333333, 0, 0.33333, 0.6666666, 0.33333, 0, 0.66666, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]

    classifier.fit(X_train_train, y_train_train)
    posteriors = classifier.posteriors
    assert np.isclose(classifier.priors['on time'], 0.7) and np.isclose(classifier.priors['late'], 0.1) and np.isclose(classifier.priors['very late'], 0.15) and np.isclose(classifier.priors['cancelled'], 0.05)
    assert test_nested_dict_values_match(classifier.posteriors, expected_values)
    

def test_nested_dict_values_match(dictionary, expected_values):
    """
    test that each value in the nested dictionary matches the expected value."""
    
    outer_keys = ["on time", "late", "very late", "cancelled"]
    # Inner dictionary keys (attributes)
    inner_keys = [
        "att1=weekday", "att1=saturday", "att1=holiday", "att1=sunday",
        "att2=spring", "att2=winter", "att2=summer", "att2=autumn",
        "att3=none", "att3=high", "att3=normal",
        "att4=none", "att4=slight", "att4=heavy"
    ]
    index = -1
    for outer_key in outer_keys:
        for inner_key in inner_keys:
            index += 1
            actual_value = dictionary[outer_key][inner_key]
            if index < len(expected_values):
                expected_value = expected_values[index]
                assert np.allclose(actual_value, expected_value)
    return True


test_naive_bayes_classifier_fit()
test_naive_bayes_classifier_fit()
def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test = [[1, 5]]
    y_test = [["yes"]]

    classifier = MyNaiveBayesClassifier()
    classifier.fit(X_train_inclass_example, y_train_inclass_example)
    preds = classifier.predict(X_test, y_train_inclass_example)
    assert preds == ["yes"]

    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]

    classifier.fit(X_train_iphone, y_train_iphone)
    preds = classifier.predict(X_test, y_train_iphone)
    assert preds == ['yes', 'no']

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test = [["weekday", "winter", "high", "heavy"]]

    classifier.fit(X_train_train, y_train_train)
    preds = classifier.predict(X_test, y_train_train)
    assert preds == ['very late']

def test_simple_linear_regression_classifier_fit():
    regressor = MySimpleLinearRegressor()
    classifier = MySimpleLinearRegressionClassifier(myutils.discretizer, regressor)

    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    # y = 2x + some noise
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    dlist = [[1, 3], [3, 2]]
    myutils.convert_to_1D(dlist)

    classifier.fit(X_train, y_train)

    # convert X_train to 1D list because linregress takes 2 1D lists
    X_train = myutils.convert_to_1D(X_train)
    slope, intercept, r, p, std_err = stats.linregress(X_train, y_train)

    assert np.isclose(regressor.slope, slope)
    assert np.isclose(regressor.intercept, intercept)

def test_simple_linear_regression_classifier_predict():
    np.random.seed(0)
    regressor = MySimpleLinearRegressor()
    classifier = MySimpleLinearRegressionClassifier(myutils.discretizer, regressor)
    X_train = [[val] for val in list(range(0, 6))]
    y_train = [row[0] * 2.5 + np.random.normal(0, 25) for row in X_train]
    X_test = [[3, 15]]

    X_train2 = [[val] for val in list(range(0, 4))]
    y_train2 = [row[0] * 28 + np.random.normal(0, 150) for row in X_train2]
    X_test2 = [[82, 76], [4, 800]]

    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    classifier.fit(X_train2, y_train2)
    pred2 = classifier.predict(X_test2)

    # desk calculation
    assert pred == ['low']
    assert pred2 == ['high', 'low']

def test_kneighbors_classifier_kneighbors():
    
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test_class_example1 = [[0.33, 1]]

    X_train_class_example2 = [[3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_class_example2 = [[2, 3]]

    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    X_test_bramer_example = [[9.1, 11]]

    knn_classifier = MyKNeighborsClassifier()

    knn_classifier.fit(X_train_class_example1, y_train_class_example1)
    distances, neighbor_indices = knn_classifier.kneighbors(X_test_class_example1)
    
    # assert against our desk check (1.05304 instead of 1.053 because isclose is picky?)
    assert np.isclose(distances[0][0], 0.67) and np.isclose(distances[0][1], 1) and np.isclose(distances[0][2], 1.05304)

    knn_classifier.fit(X_train_class_example2, y_train_class_example2)
    distances, neighbor_indices = knn_classifier.kneighbors(X_test_class_example2)

    assert np.isclose(distances[0][0], 1.4142) and np.isclose(distances[0][1], 1.4142) and np.isclose(distances[0][2], 2)

    knn_classifier = MyKNeighborsClassifier(5)
    knn_classifier.fit(X_train_bramer_example, y_train_bramer_example)
    distances, neighbor_indices = knn_classifier.kneighbors(X_test_bramer_example)
    print('distances', distances)

    assert np.isclose(distances[0][0], 0.60827) and np.isclose(distances[0][1], 1.236931) and np.isclose(distances[0][2], 2.20227) and np.isclose(distances[0][3], 2.80178) and np.isclose(distances[0][4], 2.91547)

def test_kneighbors_classifier_predict():
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test_class_example1 = [[0.33, 1]]

    knn_classifier = MyKNeighborsClassifier()
    knn_classifier.fit(X_train_class_example1, y_train_class_example1)
    pred = knn_classifier.predict(X_test_class_example1)

    assert pred == ['good']

def test_dummy_classifier_fit():
    X_train = [[1, 7], [3, 2]]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))

    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_train, y_train)
    assert dummy_classifier.most_common_label == "yes"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_classifier.fit(X_train, y_train)
    assert dummy_classifier.most_common_label == "no"

    y_train = list(np.random.choice(["very spicy", "spicy", "mild"], 100, replace=True, p=[0.6, 0.1, 0.3]))
    dummy_classifier.fit(X_train, y_train)
    assert dummy_classifier.most_common_label == "very spicy"

def test_dummy_classifier_predict():
    X_train1 = [[1, 7], [3, 2]]
    y_train1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_test1 = [[90, 62]]

    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_train1, y_train1)
    pred = dummy_classifier.predict(X_test1)
    assert pred == ["yes"]

    X_train2 = [[3, 2], [8, 4], [9, 87]]
    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_test2 = [[90, 47], [7.5, 59]]
    dummy_classifier.fit(X_train2, y_train2)
    pred = dummy_classifier.predict(X_test2)
    assert pred == ["no", "no"]

    X_train3 = [[87, 37], [43, 900]]
    y_train3 = list(np.random.choice(["very spicy", "spicy", "mild"], 100, replace=True, p=[0.6, 0.1, 0.3]))
    X_test3 = [[32, 3]]
    dummy_classifier.fit(X_train3, y_train3)
    pred = dummy_classifier.predict(X_test3)
    assert pred == ["very spicy"]