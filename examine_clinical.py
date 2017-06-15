from pandas import read_csv

f = open('../data/clinical/Clinical/nationwidechildrens.org_clinical_patient_brca.txt')
data = read_csv(f, sep='\t')

interesting_phenotypes = [item[:-1] for item in open('../notes/interesting.txt').readlines()]

# print interesting_phenotypes


print 'Phenotype\tCDE_ID\tMissing\tPercent Missing\tPercent Not Applicable'
missing_string = '[Not Available]'
app_string = '[Not Applicable]'
for phenotype in interesting_phenotypes:
    column = data[phenotype]
    print len(column)
    if phenotype == 'history_other_malignancy':
        yeses = [item for item in column if item == 'Yes']
        # print "YES FOR OTHER MALIGNANCY: ", str(len(yeses))

    if phenotype == 'last_contact_days_to':
        num_gt3 = len([item for item in column[2:] if item != missing_string and int(item) >= 1095])
        print num_gt3, ' subjects with follow up of more than 3 years.'

    cde_id = column[1]
    total = len(column)
    missing = len([item for item in column if item == missing_string])
    not_app = len([item for item in column if item == app_string])
    percent_missing = float(missing) / float(total)
    percent_not_app = float(not_app) / float(total)
    print phenotype + '\t' + cde_id + '\t' + str(missing) + '\t' + str(percent_missing) + '\t' + str(percent_not_app)


rows_gt3 = data.loc[data['last_contact_days_to'] != missing_string and data['last_contact_days_to'] >= 1095]
import pdb; pdb.set_trace()
