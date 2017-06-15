import pandas as pd
from pickle import load

clinical_matrix  = pd.read_csv('clinical/Clinical/nationwidechildrens.org_clinical_patient_brca.txt', sep='\t')
print clinical_matrix.columns
ids = load(open('common_ids.dump'))
for sample_id in ids:
    if sample_id.endswith('11'):  # filter down to samples that have tumors
        continue
    sample_clinical = clinical_matrix.loc[clinical_matrix['bcr_patient_barcode'] == sample_id[:-3]]
    # process survival data



import pdb; pdb.set_trace()
