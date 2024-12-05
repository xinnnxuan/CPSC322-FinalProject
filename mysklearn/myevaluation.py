"""Programmer: Lindsey Bodenbender
Class: CPSC 322 Fall 2024
Programming Assignment #7
11/7/2024

Description: This program contains methods for various sampling methods and evaluations"""

from mysklearn import myutils
import numpy as np # use numpy's random number generation
import random
from mysklearn import myutils
from mysklearn.myclassifiers import MyNaiveBayesClassifier
from collections import defaultdict

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    tp, fp = myutils.calc_tp_fp(y_true, y_pred, labels, pos_label)

    if tp + fp == 0:
        precision = 0
        return precision
    
    precision = tp / (tp + fp)
    
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    tp, fn = myutils.calc_tp_fn(y_true, y_pred, labels, pos_label)
    if tp + fn == 0:
        recall = 0
        return recall
    recall = tp / (tp + fn)

    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall == 0:
        f1 = 0
        return f1
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def create_matrix(X, y, k, classifier, totals=False):
    labels = [i for i in range(1, 11)]
    # y_true_all and y_pred_all is y_true/y_pred for all folds
    mean_accuracy, mean_error_rate, y_true_all, y_pred_all = cross_val_predict_return_ytrue_pred(X, y, k, classifier)
    # discretize before creating matrix
    y_true_all = myutils.mpg_discretizer_list(y_true_all)
    y_pred_all = myutils.mpg_discretizer_list(y_pred_all)
    if totals:
        matrix = confusion_matrix_with_totals(y_true_all, y_pred_all, labels)
    else:
        matrix = confusion_matrix(y_true_all, y_pred_all, labels) 

    return matrix

def random_subsample(X, y, k, classifier, stratify=False):
    """Performs train_test_split on k folds"""
    mean_error_rate = 0
    accuracy_scores = []
    error_rates = []

    for i in range(k):
        # split data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=True)            
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y)
        # train classifier
        classifier.fit(X_train, y_train)
        if isinstance(classifier, MyNaiveBayesClassifier):
            preds = classifier.predict(X_test, y_train)
        else:
            preds = classifier.predict(X_test)

        # calculate accuracy and error
        accuracy = accuracy_score(y_test, preds)
        error = (1 - accuracy)
        accuracy_scores.append(accuracy)
        error_rates.append(error)
    # calculate mean accuracies and error rates
    mean_accuracy = sum(accuracy_scores) / k
    mean_error_rate = sum(error_rates) / k

    return mean_accuracy, mean_error_rate

def cross_val_predict(X, y, k, classifier):
    accuracy_scores = []
    error_rates = []
    
    # split data
    folds = kfold_split(X, n_splits=k)
    for train_indexes, test_indexes in folds:
        # split the data into training and testing sets based on indices
        X_train = [X[i] for i in train_indexes]
        y_train = [y[i] for i in train_indexes]
        X_test = [X[i] for i in test_indexes]
        y_test = [y[i] for i in test_indexes]

        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
      
        accuracy = accuracy_score(y_test, pred)
        error_rate = 1 - accuracy
        accuracy_scores.append(accuracy)
        error_rates.append(error_rate)

    # calculate mean accuracy and error rates
    mean_accuracy = sum(accuracy_scores) / k
    mean_error_rate = sum(error_rates) / k

    return mean_accuracy, mean_error_rate

def cross_val_predict_return_ytrue_pred(X, y, k, classifier):
    accuracy_scores = []
    error_rates = []
    y_true_all = []
    y_pred_all = []
    
    # split data
    folds = kfold_split(X, n_splits=k)
    for train_indexes, test_indexes in folds:
        # split the data into training and testing sets based on indices
        X_train = [X[i] for i in train_indexes]
        y_train = [y[i] for i in train_indexes]
        X_test = [X[i] for i in test_indexes]
        y_test = [y[i] for i in test_indexes]

        # train classifier and predict
        classifier.fit(X_train, y_train)
        
        if isinstance(classifier, MyNaiveBayesClassifier):
            pred = classifier.predict(X_test, y_train)
        else:
            pred = classifier.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(pred)

        accuracy = accuracy_score(y_test, pred)
        error_rate = 1 - accuracy
        accuracy_scores.append(accuracy)
        error_rates.append(error_rate)

    # calculate mean accuracy and error rates
    mean_accuracy = sum(accuracy_scores) / k
    mean_error_rate = sum(error_rates) / k

    return mean_accuracy, mean_error_rate, y_true_all, y_pred_all

def cross_val_predict_2(X, y, k, classifier):
    accuracy_scores = []
    error_rates = []
    y_true_all = []
    y_pred_all = []
    
    # split data
    folds = kfold_split2(X, n_splits=k)
    for train_indexes, test_indexes in folds:
        # split the data into training and testing sets based on indices
        X_train = [X[i] for i in train_indexes]
        y_train = [y[i] for i in train_indexes]
        X_test = [X[i] for i in test_indexes]
        y_test = [y[i] for i in test_indexes]

        # train classifier and predict
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
      
        y_true_all.extend(y_test)
        y_pred_all.extend(pred)

        accuracy = accuracy_score(y_test, pred)
        error_rate = 1 - accuracy
        accuracy_scores.append(accuracy)
        error_rates.append(error_rate)

    # calculate mean accuracy and error rates
    mean_accuracy = sum(accuracy_scores) / k
    mean_error_rate = sum(error_rates) / k

    return mean_accuracy, mean_error_rate, y_true_all, y_pred_all

def bootstrap_method(X, y, k, classifier):
    accuracy_scores = []
    error_rates = []
    random_state = 0
    for i in range(k):
        X_test, X_train, y_out_of_bag, y_sample = bootstrap_sample(X, y, k, random_state)
        random_state += 1
        classifier.fit(X_train, y_sample)
        pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_out_of_bag, pred)
        error_rate = 1 - accuracy
        accuracy_scores.append(accuracy)
        error_rates.append(error_rate)
    
    # calculate mean accuracy and error rates
    mean_accuracy = sum(accuracy_scores) / k
    mean_error_rate = sum(error_rates) / k
    
    return mean_accuracy, mean_error_rate

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True, stratify=False):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state == None:
        random_state = 0
    
    if shuffle:
        X = X.copy()
        y = y.copy()
        myutils.randomize_in_place(X, y)
    
    if stratify:
        # create dictionary to store indices for each class
        stratified_indices = defaultdict(list)
        for index, label in enumerate(y):
            stratified_indices[label].append(index)
        # split within each class
        X_train, X_test, y_train, y_test = [], [], [], []
        for label, indices in stratified_indices.items():
            # calculate the number of samples in the test set for this class
            num_test_samples = int(test_size * len(indices))

            # split the indices
            test_indices = indices[:num_test_samples]
            train_indices = indices[num_test_samples:]

            # append splits to respective lists
            X_train.extend(X[train_indices[i]] for i in range(len(train_indices) - 1))
            X_test.extend(X[test_indices[i]] for i in range(len(test_indices) - 1))
            y_train.extend(y[train_indices[i]] for i in range(len(train_indices) - 1))
            y_test.extend(y[test_indices[i]] for i in range(len(test_indices) - 1))

    else:
        # 2:1 split
        if isinstance(test_size, float):
            starting_test_index = int(test_size * len(X)) + 1
            X_train = X[:len(X) - starting_test_index]
            X_test = X[starting_test_index + 1:]
            y_train = y[:len(X) - starting_test_index]
            y_test = y[starting_test_index + 1:]
        if isinstance(test_size, int):
            starting_test_index = (len(X) - test_size)
            X_train = X[:starting_test_index]
            X_test = X[starting_test_index:]
            y_train = y[:starting_test_index]
            y_test = y[starting_test_index:]

    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    folds = []
    divided_data = []

    # size of first n_samples % n_splits
    num_size_1_indexes = len(X) % n_splits
    # size n_samples // n_splits + 1
    size_1 = int((len(X) / n_splits)) + 1
    # other folds sizes = n_samples // n_splits
    size_2 = int(len(X) / n_splits)

    size_1_indexes = []
    size_2_indexes = []
    index = 0

    X_indexes = [i for i, val in enumerate(X)]

    if shuffle:
        # X_indexes = X_indexes.copy()
        myutils.randomize_in_place(X_indexes)


    # split the data into properly sized
    if num_size_1_indexes > 0:
        for i in range(num_size_1_indexes):
            fold = []
            for _ in range(size_1):
                fold.append(X_indexes[index])
                index += 1 # increment index each time an item is added to a fold
            size_1_indexes.append(fold)

        for i in range(n_splits - num_size_1_indexes):
            fold = []
            for _ in range(size_2):
                fold.append(X_indexes[index])
                index += 1
            size_2_indexes.append(fold)
    else:
        for i in range(n_splits):
            fold = []
            for _ in range(size_2):
                fold.append(X_indexes[index])
                index += 1
            size_2_indexes.append(fold)

    divided_data = []
    for i in size_1_indexes:
        divided_data.append(i)
    for i in size_2_indexes:
        divided_data.append(i)

    # use the k - 1 approach to create 5 folds
    for i in range(n_splits):
    # pass in the index of the test fold
        train_indexes, test_indexes = myutils.separate_data(divided_data, i)
        
        train_indexes = myutils.convert_to_1D(train_indexes)
        test_indexes = myutils.convert_to_1D(test_indexes)
        
        folds.append((train_indexes, test_indexes))

    return folds

def kfold_split2(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    folds = []
    divided_data = []

    # size of first n_samples % n_splits
    num_size_1_indexes = len(X) % n_splits
    # size n_samples // n_splits + 1
    size_1 = int((len(X) / n_splits)) + 1
    # other folds sizes = n_samples // n_splits
    size_2 = int(len(X) / n_splits)

    size_1_indexes = []
    size_2_indexes = []
    index = 0

    X_indexes = [i for i, val in enumerate(X)]

    if shuffle:
        # X_indexes = X_indexes.copy()
        myutils.randomize_in_place(X_indexes)


    # split the data into properly sized
    if num_size_1_indexes > 0:
        for i in range(num_size_1_indexes):
            fold = []
            for _ in range(size_1):
                fold.append(X_indexes[index])
                index += 1 # increment index each time an item is added to a fold
            size_1_indexes.append(fold)

        for i in range(n_splits - num_size_1_indexes):
            fold = []
            for _ in range(size_2):
                fold.append(X_indexes[index])
                index += 1
            size_2_indexes.append(fold)
    else:
        for i in range(n_splits):
            fold = []
            for _ in range(size_2):
                fold.append(X_indexes[index])
                index += 1
            size_2_indexes.append(fold)

    divided_data = []
    for i in size_1_indexes:
        divided_data.append(i)
    for i in size_2_indexes:
        divided_data.append(i)

    # use the k - 1 approach to create 5 folds
    for i in range(n_splits):
    # pass in the index of the test fold
        train_indexes, test_indexes = myutils.separate_data_convert_to_1D(divided_data, i)
        
        train_indexes = myutils.convert_to_1D(train_indexes)
        test_indexes = myutils.convert_to_1D(test_indexes)
        
        folds.append((train_indexes, test_indexes))

    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    return # TODO fix this

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """

    X_sample = []
    X_out_of_bag = []
    if y is not None:
        y_sample = []
        y_out_of_bag = []

    if random_state == None:
        random_state = 0
    if n_samples == None:
        n_samples = len(X)
    np.random.seed(random_state)

    # get random samples
    for i in range(n_samples):
        rand_index = np.random.randint(0, len(X))
        X_sample.append(X[rand_index])
        if y is not None:
            y_sample.append(y[rand_index])
    # get out of bag samples
    for i, val in enumerate(X):
        if val not in X_sample:
            X_out_of_bag.append(val)
            if y is not None:
                y_out_of_bag.append(y[X.index(val)])

    if y is not None:
        return X_sample, X_out_of_bag, y_sample, y_out_of_bag
    else:
        return X_sample, X_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    # create matrix full of 0s
    for i, val in enumerate(labels):
        row = [0 for _ in range(len(labels))]
        matrix.append(row)

    for i, val in enumerate(y_true):
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1

    return matrix

def confusion_matrix_with_totals(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    # create matrix full of 0s
    for i, val in enumerate(labels):
        row = [0 for _ in range(len(labels))]
        matrix.append(row)

    for i, val in enumerate(y_true):
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1

    # calculate and append row and col totals
    row_totals, col_totals = calculate_totals(matrix, labels)
    for i, row in enumerate(matrix):
        row.append(row_totals[i])
    matrix.append(col_totals)

    # append total
    total = sum(row_totals)
    matrix[len(matrix) - 1].append(total)

    return matrix

def calculate_totals(matrix, labels):
    row_totals = []
    col_totals = [0 for _ in range(len(labels))]
    # calculate row totals
    for row in matrix:
        row_totals.append(sum(row))
    # calculate column totals
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            col_totals[j] += row[j]
    return row_totals, col_totals


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correctly_classified = 0
    for i, val in enumerate(y_pred):
        if y_true[i] == y_pred[i]:
            correctly_classified += 1
    if normalize == True:
        correctly_classified = correctly_classified / (len(y_true))

    return correctly_classified
