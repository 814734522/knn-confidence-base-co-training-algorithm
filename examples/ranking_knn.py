"""
    Running KNN Recommenders [Item Recommendation]

    - Cross Validation
    - Simple

"""

from caserec.recommenders.item_recommendation.user_attribute_knn import UserAttributeKNN
from caserec.recommenders.item_recommendation.item_attribute_knn import ItemAttributeKNN
from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.utils.cross_validation import CrossValidation

db = '../../../datasets/ml-1m/ratings.csv'
folds_path = '../../../datasets/ml-1m/'

metadata_item = '../../../datasets/ml-1m/metadata_movies.dat'
sm_item = '../../../datasets/ml-1m/sim_item.dat'
metadata_user = '../../../datasets/ml-1m/metadata_user.csv'
sm_user = '../../../datasets/ml-1m/sim_user.dat'

tr = '../../../datasets/ml-1m/folds/0/train.dat'
te = '../../../datasets/ml-1m/folds/0/test.dat'

"""

    UserKNN

"""

# # Cross Validation
# recommender = UserKNN(as_binary=True)
#
# CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()
#
# # Simple
# UserKNN(tr, te, as_binary=True).compute()
# UserAttributeKNN(tr, te, metadata_file=metadata_user).compute()
# UserAttributeKNN(tr, te, similarity_file=sm_user).compute()

"""

    ItemKNN

"""

# # Cross Validation
recommender = ItemKNN(as_binary=True)
#
CrossValidation(input_file=db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=5).compute()
#
# # Simple
# ItemKNN(tr, te, as_binary=True).compute()
ItemAttributeKNN(tr, te, metadata_file=metadata_item).compute()
# ItemAttributeKNN(tr, te, similarity_file=sm_item).compute()
