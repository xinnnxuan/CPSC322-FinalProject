"""Programmer: Lindsey Bodenbender
Class: CPSC 322 Fall 2024
11/18/2024

Description: Classes to represent several different classifiers"""

import operator
from mysklearn import myutils
import numpy as np
import random

class MyRandomForestClassifier:
    """Represents a random forest classifier
    
    Attributes: 
        X_train (list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        N (int): The number of decision trees to generate.
        M (int): The number of most accurate ("best") decision trees to use for majority voting
        F (int): The number of randomly selected attributes to consider at each node.
        test_set (list of list of obj): The stratified test set (1/3 of the dataset).
        remainder_set(list of list of obj): The remaineder set (2/3 of the dataset).
        decision_trees (list of MyDecisionTreeClassifier): The decision trees in the forest
        selected_trees (list of MyDecisionTreeClassifier): The M most accurate decsion trees.
        test_predictions (list of obj): Predictions for the test set 
    """
    def __init__(self):
        """Initializer for MyRandomForestClassifier"""
        self.X_train = None
        self.y_train = None
        self.test_set = None
        self.remainder_set = None
        self.N = None # number of trees
        self.M = None # number of "best" trees
        self.F = None # number of random atts to consider at each split
        self.selected_trees = [] # stores the M best trees
    
    def fit(self, X, y, N, M, F):
        selected_trees = []
        trees = []
        accuracy_scores = []
        # generate random stratified test set
        self.X_train, X_test, self.y_train, y_test = myutils.train_test_split(X, y, stratify=True) # X_train, X_test, y_train, y_test
        self.remainder_set = (self.X_train, self.y_train)
        self.test_set = (X_test, y_test)
        self.N = N
        self.M = M
        self.F = F

        # use 2/3 of the set to generate N random decision trees using bootstrapping
        for i in range(self.N):
            X_train = self.remainder_set[0]
            y_train = self.remainder_set[1]

            X_sample, X_out_of_bag, y_sample, y_out_of_bag = myutils.bootstrap_sample(X_train, y_train)
            decision_tree_classifier = MyDecisionTreeClassifier()
            decision_tree_classifier.fit(X_sample, y_sample, F=F)
            trees.append(decision_tree_classifier)

        # have each tree predict
        for tree in trees:
            y_pred = tree.predict(X_out_of_bag)
            # calculate the accuracy score
            accuracy_score = myutils.accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy_score)
        
        # get the highest accuracy_scores
        accuracy_scores.sort()

        # return a list of the M most accurate ones
        self.selected_trees = selected_trees
        return self.selected_trees

    def predict(self, X_test):
        pass


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.attribute_domains = None
        self.tree = None
        self.F = None

    def fit(self, X_train, y_train, F=None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        # extract header and attribute_domains
        self.header, self.attribute_domains = myutils.extract_header_att_domains(X_train, y_train)
        self.F = F
        if not isinstance(X_train[0], list):
            X_train = myutils.convert_to_2D(X_train)
        # lets stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))] # each instance w its class label on the end
        # make a copy of header, bc python pass by object reference
        # and tdidt will be removing attributes from available_attributes
        available_attributes = self.header.copy()
        tree = self.tdidt(train, available_attributes)
        # your unit test will assert tree == interview_tree_solution (order matters!)
        self.tree = tree
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            prediction = self.predict_instance(instance, self.tree)
            y_predicted.append(prediction)
        return y_predicted
    
    def predict_instance(self, instance, tree):
        """Predicts the class label for a single test instance by traversing the tree"""
        # check if the current node is a leaf
        if tree[0] == "Leaf":
            return tree[1]
        # otherwise, it's an attribute node
        attribute = tree[1]
        attribute_index = self.header.index(attribute)  # index of the attribute in the instance
        instance_value = instance[attribute_index]  # value of the attribute in the test instance

        # traverse the branches
        for branch in tree[2:]:
            if branch[0] == "Value" and branch[1] == instance_value:
                return self.predict_instance(instance, branch[2])  # recurse into the subtree

        # if no matching branch is found, handle it
        return None
    
    def partition_instances(self, instances, attribute):
        # this is group by attribute domain (not values of attribute in instances)
        # lets use dictionaries
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions
    
    def tdidt(self, current_instances, available_attributes):
        # select a subset of attributes if F is specified
        if self.F and self.F < len(available_attributes):
            candidate_attributes = random.sample(available_attributes, self.F)
        else:
            candidate_attributes = available_attributes.copy()
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, candidate_attributes)
        available_attributes.remove(split_attribute) # can't split on this attribute again
        # in this subtree
        tree = ["Attribute", split_attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same 
            # => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                leaf = self.make_leaf(att_partition, current_instances, split_attribute)
                value_subtree.append(leaf)
            #    CASE 2: no more attributes to select (clash) 
            # => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                leaf = self.make_leaf_case_2(att_partition, current_instances, split_attribute)
                value_subtree.append(leaf)
            #    CASE 3: no more instances to partition (empty partition) 
            # => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                majority_vote = most_frequent([instance[-1] for instance in current_instances])
                num = sum(1 for instance in current_instances if instance[-1] == majority_vote)
                den = len(current_instances)
                leaf = ["Leaf", majority_vote, num, den]
                value_subtree.append(leaf)
            else:
                # none of the base cases were true recurse!
                # value_subtree.append(subtree)
                value_subtree.append(self.tdidt(att_partition, available_attributes))
            tree.append(value_subtree)
        return tree
    
    def select_attribute(self, instances, attributes):
        # for each available attribute
        #       for each value in the attribute's domain
        #           calculate the entropy for the value's partition
        #       calculate the weighted average for the partition entropies
        # select the attribute with the smallest enew entropy
        # for now, select an attribute randomly
        class_labels = sorted(myutils.get_unique_labels(myutils.extract_col(instances, attributes, -1)))
        partition_entropy = []
        weighted_averages = []

        for attribute in attributes:
            # partition instances based on the current attribute
            partitions = self.partition_instances(instances, attribute)
            total_instances = len(instances)
            weighted_entropy = 0
            att_domain = myutils.attribute_domains[attribute]

            for att_value, partition in partitions.items():
                partition_size = len(partition)
                # calculate the entropy for the value's partition
                counts = myutils.count_class_labels(partition, class_labels)
                for count in counts:
                    if count == 0:
                        posteriors = [0]
                    else:
                        posteriors = [count / sum(counts) for count in counts]
                entropy = myutils.compute_entropy(posteriors)
                # calculate the weighted average for the partition entropy's
                weighted_entropy += (partition_size / total_instances) * entropy
            partition_entropy.append(weighted_entropy)
            weighted_averages.append(weighted_entropy)
            # random attribute selection
            # rand_index = np.random.randint(0, len(attributes))
            # return attributes[rand_index]
        return attributes[weighted_averages.index(min(weighted_averages))]

    def make_leaf(self, att_partition, current_instances, split_attribute):
        leaf = ["Leaf", att_partition[0][-1]]
        num = len(att_partition)
        den = len(current_instances)
        leaf.append(num)
        leaf.append(den)

        return leaf

    def make_leaf_case_2(self, att_partition, current_instances, split_attribute):
        """Handle clash with majority vote leaf node"""
        col = []
        for i, val in enumerate(att_partition):
            col.append(att_partition[i][-1])
        majority_vote = most_frequent(col)
        num = 0
        for i in att_partition:
            if i[-1] == majority_vote:
                num += 1
        leaf = ["Leaf", majority_vote]
        den = len(current_instances)
        leaf.append(num)
        leaf.append(den)
        return leaf

    def all_same_class(self, instances):
        first_class = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_class:
                return False
        # get here, then all same class labels
        return True

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = self.header
        rules = []
        self.traverse_tree(self.tree, [], attribute_names, class_name, rules)
        for rule in rules:
            print(rule)

    def traverse_tree(self, node, conditions, att_names, class_name, rules):
        """Helper function to traverse tree and collect rules"""
        if node[0] == "Leaf":
            label = node[1]
            rule = "IF " + "AND ".join(conditions) + f"THEN {class_name} = {label}"
            rules.append(rule)
        elif node[0] == "Attribute":
            # at decision node recurse for each branch
            att_index = int(node[1].replace("att", "")) # extract att_index
            att_name = att_names[att_index]
            for branch in node[2:]:
                if branch[0] == "Value":
                    value = branch[1]
                    # add condition for the branch and recurse
                    new_conditions = conditions + [f"{att_name} == '{value}' "]
                    self.traverse_tree(branch[2], new_conditions, att_names, class_name, rules)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        self.posteriors = {}
        all_categories = [] # 2D
        all_categories_1D = [] # 1D list
        posteriors_2D = []
    
        # create list of class labels
        class_labels = myutils.get_unique_labels(y_train)
        class_label_counts = [0 for i in range(len(class_labels))]

        # count occurrences of each class label and divide by total instances
        for label in class_labels:
            self.priors[label] = y_train.count(label) / len(y_train)

        # initialize posteriors for each class label as nested dictionaries
        for label in class_labels:
            self.posteriors[label] = {}
        
        # count the occurrences of each class label
        for i, label in enumerate(class_labels):
            for val in y_train:
                if val == label:
                    class_label_counts[class_labels.index(label)] += 1 
    
        # create list of all possible categories for each attribute
        for i, val in enumerate(X_train[0]):
            col = myutils.extract_col(X_train, class_labels, i)
            categories = myutils.get_unique_labels(col)
            all_categories.append(categories)

        for i, row in enumerate(all_categories):
            for j, cat in enumerate(row):
                all_categories_1D.append(all_categories[i][j]) 

        # count how many categories there are
        num_categories = 0
        for row in all_categories:
            for j in row:
                num_categories += 1

        # create posteriors list of 0s 
        for i in range(len(class_labels)):
            row = []
            for i in range(num_categories):
                row.append(0)
            posteriors_2D.append(row)
        
        index = -1
        for row in all_categories:
            for i, val in enumerate(row):
                index += 1
                for j, instance in enumerate(X_train):
                    if instance[all_categories.index(row)] == val:
                        posteriors_2D[class_labels.index(y_train[j])][index] += 1
        for i, row in enumerate(posteriors_2D):
            for j, val in enumerate(row):
                val = val / class_label_counts[i]
                posteriors_2D[i][j] = val

        self.posteriors = myutils.create_posterior_dict(class_labels, posteriors_2D, all_categories)

    def predict(self, X_test, y_train):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        to_calculate = []

        class_labels = myutils.get_unique_labels(y_train)

        # get the posteriors matching the values in X_test
        for i, row in enumerate(X_test):
            for label in class_labels:
                temp_row = []
                for j, val in enumerate(row):
                    # temp_row = []
                    key = f'att{j + 1}={val}'
                    # check that the key exists in posteriors
                    if key in self.posteriors[label]:
                        posterior = self.posteriors[label][key]
                    else:
                        posterior = 1e-6
                    temp_row.append(posterior)
                to_calculate.append(temp_row)
        
        for i, row in enumerate(to_calculate):
            row.append(self.priors[class_labels[i % len(class_labels)]])
        
        probabilities = [np.prod(row) for row in to_calculate]
        y_predicted = myutils.predict_labels(probabilities, class_labels)
        
        return y_predicted

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        X_train = myutils.convert_to_2D(X_train)
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        X_test = myutils.convert_to_2D(X_test)
        y_numeric_predicted = self.regressor.predict(X_test)
        y_predicted = [self.discretizer(pred) for pred in y_numeric_predicted]

        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        all_distances = []
        dist_and_index = []
        distances = []
        neighbor_indices = []

        # convert X_test and X_train to 2D list if needed
        X_test = myutils.convert_to_2D(X_test)
        self.X_train = myutils.convert_to_2D_specify_num_items(self.X_train, len(X_test[0])) 

        if not isinstance(self.X_train[0][0], float) and not isinstance(self.X_train[0][0], int):
            for i, row in enumerate(self.X_train):
                dist = myutils.compute_categorical_distance(row, X_test)
                all_distances.append(dist)
        else:
            # compute distances from every X_train instance to every X_test instance
            for i, row in enumerate(self.X_train):
                dist = myutils.compute_euclidean_distance(row, X_test)
                all_distances.append(dist)
    
        # append corresponding X_train indices to distances
        index = 0
        for i in range(len(X_test)):
            row = [] 
            # for j, val in enumerate(all_distances):
            for _ in all_distances:
                for dist in _:
                    row.append((index, dist))
                index += 1
            dist_and_index.append(row)
            index = 0
        closest_dist = []
        # sort the distances based on index -1 (the distance), slice the top 3
        for i in dist_and_index:
            i.sort(key=operator.itemgetter(-1))

        # find the closest distances
        for i in dist_and_index:
            closest_dist.append(i[:self.n_neighbors])

        # separate into two lists to return
        for i, row in enumerate(closest_dist):
            temp_indices = []
            temp_dist = []
            for j, val in enumerate(row):
                temp_indices.append(val[0])
                temp_dist.append(val[1])
            neighbor_indices.append(temp_indices)
            distances.append(temp_dist)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        labels = [] # labels of the closest neighbors

        # convert X_test to 2D list if needed
        X_test = myutils.convert_to_2D(X_test) 

        # retreive the k nearest neighbors (by calling kneighbors)
        distances, neighbor_indices = self.kneighbors(X_test) 

        # get the values of the closest neighbors
        for i in X_test:
            for item in neighbor_indices:
                row = []
                for index in item:
                    row.append(self.y_train[index]) 
            labels.append(row)

        # take a majority vote on those y values to predict the classification
        for i in labels:
            pred = most_frequent(i)
            y_predicted.append(pred)

        return y_predicted

def most_frequent(list):
    return max(set(list), key=list.count)

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = most_frequent(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [self.most_common_label for i in range(len(X_test))]

        return y_predicted

