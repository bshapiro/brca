python baselines.py -m knn --data ec
python baselines.py -m knn --data e
python baselines.py -m knn --data c

python baselines.py -g pam50 -m knn --data ec
python baselines.py -g pam50 -m knn --data e

python baselines.py -m svm --data ec
python baselines.py -m svm --data e
python baselines.py -m svm --data c

python baselines.py -g pam50 -m svm --data ec
python baselines.py -g pam50 -m svm --data e

python baselines.py -m logreg --data ec
python baselines.py -m logreg --data e
python baselines.py -m logreg --data c

python baselines.py -g pam50 -m logreg --data ec
python baselines.py -g pam50 -m logreg --data e
