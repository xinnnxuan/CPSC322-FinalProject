import pickle

def save_tree(wine_header, tree):
    """Saves the decision tree classifier to a pickle file"""
    packaged_obj = (wine_header, tree)
    outfile = open('tree.p', 'wb')
    pickle.dump(packaged_obj, outfile)
    outfile.close()
    
def load_tree(filename="decision_tree.p"):
    """Loads the decision tree classifier from a pickle file"""
    with open(filename, "rb") as f:
        decision_tree_classifier = pickle.load(f)
    return decision_tree_classifier
