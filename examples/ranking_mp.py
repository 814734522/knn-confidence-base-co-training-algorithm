"""
    Running Most Popular Recommender [Item Recommendation]

    - Cross Validation
    - Simple

"""

from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.utils.cross_validation import CrossValidation

db = '../../../datasets/ml-1m/ratings.csv'
folds_path = '../../../datasets/ml-1m/'

tr = '../../../datasets/ml-1m/folds/0/train.dat'
te = '../../../datasets/ml-1m/folds/0/test.dat'

# Cross Validation
recommender = MostPopular(as_binary=True)

CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()

# Simple
MostPopular(tr, te, as_binary=True).compute()
