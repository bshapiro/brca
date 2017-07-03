
for model in svm knn neural
do

python baselines.py -m $model --data ec
python baselines.py -m $model --data e
python baselines.py -m $model --data c
python baselines.py -m $model --data r
python baselines.py -m $model --data cr
python baselines.py -m $model --data erc

python baselines.py -g pam50 -m $model --data ec
python baselines.py -g pam50 -m $model --data ecr
python baselines.py -g pam50 -m $model --data er
python baselines.py -g pam50 -m $model --data e

python baselines.py -g protein -m $model --data ec -r
python baselines.py -g protein -m $model --data ecr -r 
python baselines.py -g protein -m $model --data er -r 
python baselines.py -g protein -m $model --data e -r 

python baselines.py -g full -m $model --data ec -r
python baselines.py -g full -m $model --data ecr -r
python baselines.py -g full -m $model --data er -r
python baselines.py -g full -m $model --data e -r

done
