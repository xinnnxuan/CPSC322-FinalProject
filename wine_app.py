import pickle

def load_model():
    # unpickle header and tree in tree.p
    infile = open('tree.p', 'rb')
    header, tree = pickle.load(infile)
    infile.close()
    return header, tree

if __name__ == "__main__":
    header, tree = load_model()
    print(header)
    print(tree)