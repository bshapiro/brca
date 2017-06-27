import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def train_model(model_name, data, labels, params):
    if model_name == 'knn':
        model = train_knn(data, labels, params)
    elif model_name == 'svm':
        model = train_svm(data, labels, params)
    elif model_name == 'logreg':
        model = train_logreg(data, labels, params)
    else:
        sys.exit('Model does not exist.')
    return model


def train_knn(data, labels, params):
    neighbors = params[0]
    model = KNeighborsClassifier(neighbors, n_jobs=-1)
    model.fit(data, labels)
    return model


def train_svm(data, labels, params):
    kernel, penalty, gamma = params[0], params[1][0], params[1][1]
    model = SVC(kernel=kernel, C=penalty, gamma=gamma)
    model.fit(data, labels)
    return model


def train_logreg(data, labels, params):
    regularization_type, penalty = params[0], params[1]
    model = LogisticRegression(penalty=regularization_type, C=penalty, solver='lbfgs', multi_class='multinomial')  # multi_class option enforces softmax approach
    model.fit(data, labels)
    return model

def train_intermediate(data, labels, params):
    intermediate_labels = data[:, -8:]
    import pdb; pdb.set_trace()

