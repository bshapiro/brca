python baselines.py -m svm --data ec
python baselines.py -m svm --data e
python baselines.py -m svm --data c
python baselines.py -m svm --data r
python baselines.py -m svm --data cr
python baselines.py -m svm --data erc

python baselines.py -g pam50 -m svm --data ec
python baselines.py -g pam50 -m svm --data ecr
python baselines.py -g pam50 -m svm --data er
python baselines.py -g pam50 -m svm --data e

python baselines.py -m logreg --data r --receptor-to-subtype
python baselines.py -m logreg --data rc --receptor-to-subtype
