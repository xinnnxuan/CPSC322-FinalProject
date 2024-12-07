import pickle

def save_tree(wine_header, tree):
    """Saves the decision tree classifier to a pickle file"""
    packaged_obj = (wine_header, tree)
    outfile = open('tree.p', 'wb')
    pickle.dump(packaged_obj, outfile)
    outfile.close()
    
