import os
import fnmatch


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files, in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

files = recursive_glob('../results/baselines/', '*.txt')

for filename in files:
    configuration = (filename.split('baselines')[1]).split('results.txt')[0]
    f = open(filename)
    result = f.readlines()[-3:]
    accuracy = result[0].split()[-1]
    precision = result[1].split()[-1]
    recall = result[2].split()[-1]
    print configuration, accuracy, precision, recall
