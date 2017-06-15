from optparse import OptionParser
from helpers import *

parser = OptionParser()
parser.add_option("-e", "--exp", dest="exp_file", default="../data/bgam_exp.txt",
                  help="Location of gene expression.")
parser.add_option("-c", "--clinical", dest="clinical_file", default="../data/bgam_clinical.txt",
                  help="Location of clinical data.")
parser.add_option("-s", "--survival", dest="survival_file", default="../data/bgam_survival.txt",
                  help="Location of survival data")
parser.add_option("-m", "--model", dest="model", default="knn",
                  help="Model to use")
parser.add_option("-g", "--geneset", dest="geneset", default='full',
                  help="Geneset to use")

(options, args) = parser.parse_args()


def cross_validate(model, geneset, params):
    pass


if __name__ == "__main__":
    params = {'kNN': [1, 2, 4, 8, 16, 32]}
    results_dir = '../results/baselines/' + options.model + '/' + options.geneset
    try:
        os.makedirs(results_dir)
    except:
        print 'Directory exists.'
    data = process_data(options.clinical_file, options.exp_file,
                        options.survival_file, options.geneset)
    results_filename = results_dir + "/results.txt"
    results_f = open(results_filename, 'w')
    cross_validate(data, model, params[model], f)
