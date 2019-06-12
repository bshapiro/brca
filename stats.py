from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import sys


def train_model(model_name, data, labels, params):
    if model_name == 'knn':
        model = train_knn(data, labels, params)
    elif model_name == 'svm':
        model = train_svm(data, labels, params)
    elif model_name == 'logreg':
        model = train_logreg(data, labels, params)
    elif model_name == 'nn':
        model = train_nn(data, labels, params)
    else:
        sys.exit('Model does not exist.')
    return model


def train_knn(data, labels, params):
    neighbors = params[0]
    model = KNeighborsClassifier(neighbors, n_jobs=-1)
    model.fit(data, labels)
    return model


def train_svm(data, labels, params):
    class_weight = float(len(data)) / (2*np.bincount(np.asarray(labels, dtype='int')))  # balanced
    class_weight[1] = 10*class_weight[1]  # adjust by factor of 2
    class_weight_dict = dict(zip([0, 1], list(class_weight)))
    kernel, penalty, gamma = params[0], params[1][0], params[1][1]
    model = SVC(kernel=kernel, C=penalty, gamma=gamma, class_weight=class_weight_dict)
    model.fit(data, labels)
    return model


def train_logreg(data, labels, params):
    regularization_type, penalty = params[0], params[1]
    model = LogisticRegression(penalty=regularization_type, C=penalty, solver='lbfgs', multi_class='multinomial', class_weight='balanced')  # multi_class option enforces softmax approach
    model.fit(data, labels)
    return model


def train_nn(data, labels, params):
    alpha = params[0]
    model = MLPClassifier(hidden_layer_sizes=(256), activation='relu', alpha=alpha)
    model.fit(data, labels)
    return model


def reduction(data, phenotype, data_type):
    if data_type == 'exp':
        PCA_model = PCA(n_components=50)
        exp_pca = PCA_model.fit_transform(data)
        LDA_model = LDA()
        exp_lda = LDA_model.fit_transform(data, phenotype)
        exp_reduced = np.concatenate((exp_pca, exp_lda), 1)
        return exp_reduced
    elif data_type == 'methylation':
        pass
    else:
        return data
