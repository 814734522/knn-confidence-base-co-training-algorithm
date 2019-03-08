# coding=utf-8

from caserec.recommenders.rating_prediction.userknn import UserKNN
from caserec.recommenders.rating_prediction.base_knn import BaseKNN
from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
import numpy as np


class GetKNN(BaseKNN):
    def __init__(self, train_file=None, test_file=None, similarity_metric='cosine ', k_neighbor=None, sep=','):

        super(GetKNN, self).__init__(train_file=train_file, test_file=test_file,
                                     similarity_metric=similarity_metric, sep=sep, output_sep=',')

        self.similarity_metric = None
        self.sep = None
        self.k_neighbors = k_neighbor
        self.users_id_viewed_item = None   # 用户看过的对象
        self.knn = None
        self.su_matrix = None

    def init_model(self):
        """
        Method to initialize the model. Compute similarity matrix based on user (user x user)

        """

        super(GetKNN, self).init_model()

        self.users_id_viewed_item = {}

        # Set the value for k           设置邻近数
        if self.k_neighbors is None:
            self.k_neighbors = int(np.sqrt(len(self.users)))                # 没有近邻参数是取用户数开根

        self.su_matrix = self.compute_similarity(transpose=False)

        # Map the users which seen an item with their respective ids
        for item in self.items:
            for user in self.train_set['users_viewed_item'].get(item, []):
                self.users_id_viewed_item.setdefault(item, []).append(self.user_to_user_id[user])

    # 获得邻近集
    def getknn(self, user, item):

        knn = []
        u_id = self.user_to_user_id[user]
        su_matrix = self.compute_similarity(transpose=False, nomalize=False)    # 获得相似矩阵
        neighbor = sorted(range(len(su_matrix[u_id])), key=lambda x: -su_matrix[u_id][x], reverse=True)  # 返回降序的用户邻近id

        for u in neighbor:
            user1 = self.user_id_to_user[u]                                     # id 转为用户
            if item in self.train_set['items_seen_by_user'][user1]:
                knn.append(self.train_set['feedback'][user1][item])
        if len(knn) == 0:
            return True                                                        # 没有邻近集
        return knn                                                              # 返回前邻近用户评分










