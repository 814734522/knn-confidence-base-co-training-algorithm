"""

    Running item recommendation algorithms

"""
from caserec.recommenders.item_recommendation.bprmf import BprMF

tr = '../../../datasets/ml-1m/folds/0/train.dat'
te = '../../../datasets/ml-1m/folds/0/test.dat'


BprMF(tr, te, batch_size=30).compute()
