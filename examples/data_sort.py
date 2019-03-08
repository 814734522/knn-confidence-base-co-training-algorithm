# coding=utf-8
from caserec.utils.split_database import SplitDatabase

db = '../../../datasets/filmtrust/rating_file.txt'
dir = '../../../datasets/filmtrust/'
SplitDatabase(input_file=db, dir_folds=dir, n_splits=10).k_fold_cross_validation()