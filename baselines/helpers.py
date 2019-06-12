import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import itertools
from stats import reduction
from pyensembl import EnsemblRelease


def process_data(clinical_filename, exp_filename, survival_filename, event_time_filename, geneset, data_config, do_reduction, model_type):
    data = None
    survival = np.loadtxt(survival_filename, delimiter='\t')
    event_time = np.loadtxt(event_time_filename, delimiter='\t')
    genes = [gene_name.split('.')[0] for gene_name in open(exp_filename).readline()[:-1].split('\t')]

    if 'e' in data_config:
        exp = np.loadtxt(exp_filename, delimiter='\t', skiprows=1)
        if geneset is not 'full':
            exp = filter_genes(exp, geneset, genes)
        data = preprocess_exp(exp, survival, do_reduction)
    if 'c' in data_config:
        clinical = preprocess_clinical(np.loadtxt(clinical_filename, delimiter='\t', skiprows=1))
        if data is None:
            data = clinical[:, 0:4]
        else:
            data = np.concatenate((data, clinical[:, 0:4]), 1)
    if 'r' in data_config:
        if model_type != 'intermediate':
            clinical = preprocess_clinical(np.loadtxt(clinical_filename, delimiter='\t', skiprows=1))
        else:
            clinical = np.loadtxt(clinical_filename, delimiter='\t', skiprows=1)  # don't scale the data if it's intermediate
        if data is None:
            data = clinical[:, 4:]
        else:
            data = np.concatenate((data, clinical[:, 4:]), 1)
    import pdb; pdb.set_trace()
    return data, survival, event_time


def preprocess_exp(exp, survival, do_reduction):
    exp = np.log2(exp+1.0)
    exp = scale(exp)
    if do_reduction:
        exp = reduction(exp, survival, 'exp')
    return exp


def preprocess_clinical(clinical):
    clinical = scale(clinical)
    return clinical


def filter_genes(exp, geneset, genes):
    if geneset == 'pam50':
        indices = np.loadtxt('../data/pam50.txt', delimiter=',', dtype='int')
        exp = exp[:, indices]
    elif geneset == 'protein':
        try:
            indices = np.loadtxt('../data/protein.txt', delimiter=',', dtype='int')
        except:
            print "Could not load protein coding indices. Recomputing..."
            data = EnsemblRelease(77)
            biotypes = []
            for gene_id in genes:
                try:
                    biotype = data.gene_by_id(gene_id).biotype
                except:
                    print "Missing biotype for gene id " + gene_id + ". Skipping."
                biotypes.append(biotype)
            biotypes = np.asarray(biotypes)
            indices = np.where(biotypes == u'protein_coding')[0]
        exp = exp[:, indices]
    return exp


def filter_samples_to_subtype(data, survival, subtype_filename):
    subtype = pd.read_csv(subtype_filename, sep='\t')
    subtype_ids_long = subtype['Sample'].values.tolist()
    subtype_ids = [item[:12] for item in subtype_ids_long]
    data_ids = [item[0] for item in pd.read_csv('../data/bgam_ids_clinical.txt').values.tolist()]
    overlapping_ids = [id for id in data_ids if id in subtype_ids]
    data_indices = [data_ids.index(id) for id in overlapping_ids]
    subtype_indices = [subtype_ids.index(id) for id in overlapping_ids]
    subtypes = subtype.iloc[subtype_indices]['PAM50'].values.tolist()
    subtype_map = {'Normal': 0, 'LumA': 1, 'LumB': 2, 'Basal': 3, 'Her2': 4}
    subtypes_num = [subtype_map[item] for item in subtypes]
    if data is None:  # in case we're adding subtypes to empty data
        return None, survival[data_indices], np.asarray(subtypes_num)
    return data[data_indices, :], survival[data_indices], np.asarray(subtypes_num)


def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n)]


def construct_receptor_labels(receptor):  # WIP
    labels = itertools.product([0, 1, 2], [0, 1, 2])
    label_dict = {}
    for label in labels:
        label_dict[label] = labels.index(label)
    label_vector = []
    for row in receptor:
        label_vector.append(label_dict[row])
    return np.asarray(label_vector)


def longer_survival(prediction1, prediction2, classification=True):
    if classification:
        return prediction1 > prediction2
    else:
        return prediction1 < prediction2


def compute_ci(survival, survival_times, predictions, classification=True):
    indices = range(len(survival_times))
    pairs = itertools.combinations(indices, 2)

    concordance = 0
    permissible = 0
    for pair in pairs:
        index1 = pair[0]
        index2 = pair[1]
        survival_time1 = survival_times[index1]
        survival_time2 = survival_times[index2]
        death1 = survival[index1]
        death2 = survival[index2]
        prediction1 = predictions[index1]
        prediction2 = predictions[index2]
        cur_survival_times = [survival_time1, survival_time2]
        cur_deaths = [death1, death2]
        cur_predictions = [prediction1, prediction2]
        shortest_index = cur_survival_times.index(min(cur_survival_times))
        longest_index = [item for item in range(2) if item != shortest_index][0]
        if cur_deaths[shortest_index] == 0:
            continue  # omit
        if (survival_time1 == survival_time2) and (death1 == 0) and (death2 == 0):
            continue  # omit
        permissible += 1
        if survival_time1 != survival_time2:
            if longer_survival(cur_predictions[longest_index], cur_predictions[shortest_index], classification):
                concordance += 1
            else:
                concordance += 0.5
        if survival_time1 == survival_time2 and death1 == 1 and death2 == 1:
            if prediction1 == prediction2:
                concordance += 1
            else:
                concordance += 0.5
        elif survival_time1 == survival_time2:
            if longer_survival(cur_predictions[cur_deaths.index(0)], cur_predictions[cur_deaths.index(1)]):
                concordance += 1
            else:
                concordance += 0.5
    return float(concordance) / float(permissible)
