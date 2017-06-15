import numpy as np


def process_data(clinical_filename, exp_filename, survival_filename, geneset):
    exp = np.loadtxt(exp_filename, delimiter='\t', skiprows=1)
    clinical = np.loadtxt(clinical_filename, delimiter='\t', skiprows=1)
    survival = np.loadtxt(survival_filename, delimiter='\t')
    if geneset is not 'full':
        exp = filter_genes(exp, geneset)
    return clinical, exp, survival


def filter_genes(exp, geneset):
    if geneset == 'pam50':
        indices = np.loadtxt('../data/pam50.txt', delimiter=',', dtype='int')
        exp = exp[:, indices]
    return exp
