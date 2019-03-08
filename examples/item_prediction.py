# coding=utf-8

from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.recommenders.item_recommendation.content_based import ContentBased
from caserec.recommenders.item_recommendation.bprmf import BprMF
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.paco_recommender import PaCoRecommender
from caserec.recommenders.item_recommendation.item_attribute_knn import ItemAttributeKNN
from caserec.recommenders.item_recommendation.user_attribute_knn import UserAttributeKNN
from caserec.recommenders.item_recommendation.group_based_recommender import GroupBasedRecommender

from caserec.utils.cross_validation import CrossValidation

db = '../../../datasets/ml-100k/ratings.csv'
folds_path = '../../../datasets/ml-100k/'

tr = '../../../datasets/ml-100k/folds/0/train.dat'
te = '../../../datasets/ml-100k/folds/0/test.dat'

tr_film = '../../../datasets/filmtrust/folds1/0/train.dat'
te_film = '../../../datasets/filmtrust/folds1/0/test.dat'

metadata_item = '../../../datasets/ml-100k/metadata_movies.csv'
sm_item = '../../../datasets/ml-100k/ItemSimilarity.csv'
metadata_user = '../../../datasets/ml-100k/metadata_user1.csv'
sm_user = '../../../datasets/ml-100k/UserSimilarity.csv'

label_set1 = '../../../datasets/filmtrust/folds1/0/labeled_set_1.dat'
label_set2 = '../../../datasets/filmtrust/folds1/0/labeled_set_2.dat'
ensemble = '../../../datasets/filmtrust/folds1/0/ensemble_set.dat'

train_list = [tr]

'''单独运行'''
UserKNN(tr_film, te_film, as_similar_first=True).compute()

ItemKNN(tr_film, te_film, as_similar_first=True).compute()

# ContentBased(tr, te, similarity_file=sm_item).compute()

# BprMF(tr, te).compute()

# MostPopular(tr, te, as_binary=True).compute()

# PaCoRecommender(tr, te, as_binary=True).compute()

# ItemAttributeKNN(tr, te, similarity_file=sm_item, as_similar_first=True).compute()

# UserAttributeKNN(tr, te, similarity_file=sm_user, as_similar_first=True).compute()

# GroupBasedRecommender(train_list, te).compute()

'''交叉验证'''

recommender = UserKNN(tr, te)
# CrossValidation(db, recommender=recommender, dir_folds=folds_path, header=1, k_folds=10).compute()