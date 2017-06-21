import numpy as np


def process_data(clinical_filename, exp_filename, survival_filename, geneset, data_config):
    data = None
    if 'e' in data_config:
        exp = np.loadtxt(exp_filename, delimiter='\t', skiprows=1)
        if geneset is not 'full':
            exp = filter_genes(exp, geneset)
        data = exp
    if 'c' in data_config:
        clinical = np.loadtxt(clinical_filename, delimiter='\t', skiprows=1)
        if data is None:
            data = clinical
        else:
            data = np.concatenate((data, clinical), 1)
    survival = np.loadtxt(survival_filename, delimiter='\t')

    return data, survival


def filter_genes(exp, geneset):
    if geneset == 'pam50':
        indices = np.loadtxt('../data/pam50.txt', delimiter=',', dtype='int')
        exp = exp[:, indices]
    return exp
