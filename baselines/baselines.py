from helpers import *
from optparse import OptionParser
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from stats import *
import itertools
import os
import random

parser = OptionParser()
parser.add_option("-e", "--exp", dest="exp_file", default="../data/bgam/bgam_exp.txt",
                  help="Location of gene expression.")
parser.add_option("-c", "--clinical", dest="clinical_file", default="../data/bgam/bgam_clinical.txt",
                  help="Location of clinical data.")
parser.add_option("--phenotype", dest="phenotype_file", default="../data/bgam/bgam_survival.txt",
                  help="Location of phenotype data")
parser.add_option("--event_time", dest="event_time", default="../data/bgam/bgam_survival_time.txt",
                  help="Location of event times")
parser.add_option("--subtype", dest="subtype_file", default="../data/bgam/subtypes.txt",
                  help="Location of subtype data")
parser.add_option("-d", "--data", dest="data_config", default="e",
                  help="Training configuration to use; combinations of emcrs.")
parser.add_option("-m", "--model", dest="model", default="svm",
                  help="Model to use: knn, logreg, nn, svm, or intermediate")
parser.add_option("-g", "--geneset", dest="geneset", default='full',
                  help="Geneset to use")
parser.add_option("--use-subtypes", dest="use_subtypes", action="store_true", default=False,
                  help="Use data with subtypes only")
parser.add_option("--receptor-to-subtype", action="store_true", default=False,
                  dest="receptor_to_subtype", help="Classify subtypes from receptor status.")
parser.add_option("-r", "--reduce", dest="do_reduction", action="store_true", default=False,
                  help="Perform dimensionality reduction?")
parser.add_option("--metric", dest="metric", default="ci",
                  help="Metric to use during cross validation.")
(options, args) = parser.parse_args()


def nested_cross_validate(data, phenotype, model_type, params, results_f):
    kf1 = KFold(n_splits=5)
    outer_scores = []
    outer_loop = 0
    for outer_train, outer_test in kf1.split(data):
        results_f.write('******************************')
        results_f.write('Outer loop: ' + str(outer_loop))
        results_f.write('******************************')
        data_train = data[outer_train, :]
        phenotype_train = phenotype[outer_train]
        data_test = data[outer_test, :]
        phenotype_test = phenotype[outer_test]
        kf2 = KFold(n_splits=10)
        param_scores = []
        for param_set in params:
            inner_scores = []
            for train, test in kf2.split(data_train):
                data_inner_train = data_train[train, :]
                phenotype_inner_train = phenotype_train[train]
                data_inner_test = data_train[test, :]
                phenotype_inner_test = phenotype_train[test]
                model = train_model(model_type, data_inner_train, phenotype_inner_train, param_set)
                inner_score = model.score(data_inner_test, phenotype_inner_test)
                inner_scores.append(inner_score)
            mean_inner_score = np.mean(inner_scores)
            param_scores.append(mean_inner_score)
            results_f.write('Params: ' + str(param_set) + ', score: ' + str(mean_inner_score) + '\n')
        best_param_index = param_scores.index(max(param_scores))
        best_param_set = params[best_param_index]
        outer_loop += 1
        results_f.write('Best params: ' + str(best_param_set) + '\n')
        model = train_model(model_type, data_train, phenotype_train, best_param_set)
        outer_score = model.score(data_test, phenotype_test)
        results_f.write('Score for best params on outer loop ' + str(outer_loop) + ': ' + str(outer_score) + '\n')
        outer_scores.append(outer_score)
    results_f.write('Average score over outer loops: ' + str(np.mean(outer_scores)) + '\n')


def cross_validate(data, phenotype, event_times, model_type, params, results_f, metric):
    k = 5
    shuffled_indices = range(data.shape[0])
    random.shuffle(shuffled_indices)
    fold_indices = partition(shuffled_indices, k)
    fold_scores = []
    fold_precisions = []
    fold_recalls = []
    fold_cis = []
    for fold in range(0, k):
        test = fold_indices[fold]
        tune = fold_indices[(fold + 1) % k]
        train = [index for index in shuffled_indices if (index not in test and index not in tune)]
        data_train = data[train, :]
        phenotype_train = phenotype[train]
        data_tune = data[tune, :]
        phenotype_tune = phenotype[tune]
        event_times_tune = event_times[tune]
        data_test = data[test, :]
        phenotype_test = phenotype[test]
        event_times_test = event_times[test]
        param_scores = []
        results_f.write('****************************** + \n')
        results_f.write('FOLD ' + str(fold) + '\n')
        results_f.write('******************************' + '\n')
        for param_set in params:
            model = train_model(model_type, data_train, phenotype_train, param_set)
            if metric == 'auc':
                tmp_scores = model.decision_function(data_tune)
                score = roc_auc_score(phenotype_tune, tmp_scores)
            elif metric == 'ci':
                predictions_tune = model.predict(data_tune)
                score = compute_ci(phenotype_tune, event_times_tune, predictions_tune, classification=True)
            elif metric == 'accuracy':
                score = model.score(data_tune, phenotype_tune)
            results_f.write('Params: ' + str(param_set) + ', score: ' + str(score) + '\n')
            param_scores.append(score)
        best_param_index = param_scores.index(max(param_scores))
        best_param_set = params[best_param_index]
        results_f.write('Best params: ' + str(best_param_set) + '\n')
        model = train_model(model_type, data_train, phenotype_train, best_param_set)
        fold_score = model.score(data_test, phenotype_test)
        predictions = model.predict(data_test)
        if max(phenotype) == 1:  # only for binary tests
            fold_precisions.append(precision_score(phenotype_test, predictions))
            fold_recalls.append(recall_score(phenotype_test, predictions))
            fold_cis.append(compute_ci(phenotype_test, event_times_test, predictions, classification=True))
        fold_scores.append(fold_score)
        results_f.write('Score for best params on this fold: ' + str(fold_score) + '\n')
    average_score_over_folds = np.mean(fold_scores)
    average_precision_over_folds = np.mean(fold_precisions)
    average_recall_over_folds = np.mean(fold_recalls)
    average_ci_over_folds = np.mean(fold_cis)
    results_f.write('Average score over folds: ' + str(average_score_over_folds) + '\n')
    results_f.write('Average precision over folds: ' + str(average_precision_over_folds) + '\n')
    results_f.write('Average recall over folds: ' + str(average_recall_over_folds) + '\n')
    results_f.write('Average CI over folds: ' + str(average_ci_over_folds) + '\n')


def cross_validate_intermediate(data, phenotype, event_times, model_type, params, results_f):
    k = 10
    subtype = data[:, -1:]
    data = data[:, :-1]
    if data.shape[1] == 0:
        sys.exit('Must include data other than subtype for intermediate classification.')
    shuffled_indices = range(data.shape[0])
    random.shuffle(shuffled_indices)
    fold_indices = partition(shuffled_indices, k)
    fold_scores = []
    fold_precisions = []
    fold_recalls = []
    for fold in range(0, k):
        test = fold_indices[fold]
        tune = fold_indices[(fold + 1) % k]
        train = [index for index in shuffled_indices if (index not in test and index not in tune)]
        data_train = data[train, :]
        subtype_train = subtype[train, :]
        phenotype_train = phenotype[train]
        data_tune = data[tune, :]
        subtype_tune = subtype[tune, :]
        phenotype_tune = phenotype[tune]
        data_test = data[test, :]
        subtype_test = subtype[test, :]
        phenotype_test = phenotype[test]
        param_scores = []
        results_f.write('****************************** + \n')
        results_f.write('FOLD ' + str(fold) + '\n')
        results_f.write('******************************' + '\n')
        for param_set in params:
            logreg_params = param_set[0]
            svm_params = param_set[1]
            logreg_model = train_model('logreg', data_train, subtype_train, logreg_params)
            scores_train = logreg_model.predict_proba(data_train)
            svm_model = train_model('svm', scores_train, phenotype_train, svm_params)
            scores_tune = logreg_model.predict_proba(data_tune)
            score = svm_model.score(scores_tune, phenotype_tune)
            results_f.write('Params: ' + str(param_set) + ', score: ' + str(score) + '\n')
            param_scores.append(score)
        best_param_index = param_scores.index(max(param_scores))
        best_param_set = params[best_param_index]
        results_f.write('Best params: ' + str(best_param_set) + '\n')
        logreg_model = train_model('logreg', data_train, subtype_train, best_param_set[0])
        scores_train = logreg_model.predict_proba(data_train)
        svm_model = train_model('svm', scores_train, phenotype_train, best_param_set[1])
        scores_test = logreg_model.predict_proba(data_test)
        fold_score = svm_model.score(scores_test, phenotype_test)
        predictions = svm_model.predict(scores_test)
        fold_precisions.append(precision_score(phenotype_test, predictions))
        fold_recalls.append(recall_score(phenotype_test, predictions))
        fold_scores.append(fold_score)
        results_f.write('Score for best params on this fold: ' + str(fold_score) + '\n')
    average_score_over_folds = np.mean(fold_scores)
    average_precision_over_folds = np.mean(fold_precisions)
    average_recall_over_folds = np.mean(fold_recalls)
    results_f.write('Average score over folds: ' + str(average_score_over_folds) + '\n')
    results_f.write('Average precision over folds: ' + str(average_precision_over_folds) + '\n')
    results_f.write('Average recall over folds: ' + str(average_recall_over_folds) + '\n')


if __name__ == "__main__":
    ############### EXCEPTIONS ################
    if options.model == 'intermediate':
        if 's' not in options.data_config:
            sys.exit("Can't run intermediate classification without including subtype in the data.")

    ############### FILE CONFIGURATION ################
    results_dir = '../results/baselines/' + options.model + '/' + options.data_config + '/'
    if options.receptor_to_subtype:
        results_dir += 'receptor_to_subtype/'
    if options.do_reduction:
        results_dir += 'reduced/'
    if 'e' in options.data_config:
        results_dir += options.geneset
    try:
        os.makedirs(results_dir)
    except:
        print "Directory already exists."

    ########## PROCESS & FILTER  DATA ##############
    data, phenotype, event_times = process_data(options.clinical_file, options.exp_file,
                                                options.phenotype_file, options.event_time,
                                                options.geneset, options.data_config,
                                                options.do_reduction, options.model)

    if options.use_subtypes or options.receptor_to_subtype or 's' in options.data_config or options.model is 'intermediate':
        data, phenotype, subtypes = filter_samples_to_subtype(data, phenotype, options.subtype_file)
        if 's' in options.data_config or options.model is 'intermediate':
            if data is not None:
                data = np.concatenate((data, np.reshape(subtypes, (subtypes.shape[0], 1))), 1)
            else:
                data = np.reshape(subtypes, (subtypes.shape[0], 1))

    if options.receptor_to_subtype:
        phenotype = subtypes

    ########## PARAMETER SETUP ##############
    params = {'knn': [[1], [2], [4], [8], [16], [32]],
              'svm': zip(['rbf']*144, list(itertools.product([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]))),
              'logreg': zip(['l2']*12, [0.000000000000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000000000]*1),
              'nn': [[i] for i in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]]}
    params['intermediate'] = list(itertools.product(params['logreg'], params['svm']))

    ########## RUN TASKS ##############
    results_filename = results_dir + "/results.txt"
    results_f = open(results_filename, 'w')
    model_type = options.model
    if model_type != 'intermediate':
        cross_validate(data, phenotype, event_times, model_type, params[model_type], results_f, options.metric)
    else:
        cross_validate_intermediate(data, phenotype, model_type, params[model_type], results_f)
