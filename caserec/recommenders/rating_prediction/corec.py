

from multiprocessing.pool import Pool
from random import shuffle
import numpy as np
import os
from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
from caserec.recommenders.rating_prediction.svdplusplus import SVDPlusPlus
from caserec.recommenders.rating_prediction.userknn import UserKNN
from caserec.utils.process_data import ReadFile
from caserec.utils.extra_functions import ComputeBui


class ECoRec(object):
    def __init__(self, train_file=None, test_file=None, recommenders=(1, 2), confidence_measure=None,
                 number_sample=None, m=None, sep=',', ensemble_method=False):

        """

        (E)CoRec for rating prediction

        This algorithm is based on a co-training approach, named (E)CoRec, that drives two or more recommenders to
        agree with each others’ predictions to generate their own. The output of this algorithm is n (where n is the
        number of used recommenders) new enriched user x item matrix, which can be used as new training sets.

        Usage::

            >> ECoRec(tr, te, number_sample=10, confidence_measure='su').compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param recommenders: Recommenders in which should be used as regressors to predict unlabeled sets. At least 2
        should be selected. Options:
            - 1: Item KNN
            - 2: User KNN
            - 3: Matrix Factorization
            - 4: SVD++
        :type recommenders: tuple, default (1,2)

        :param confidence_measure: Confidence Measure to calculate the precision of the predicted sample. Options:
               - 'pc': proposed metric in the articles(boosting)
               - 'vi': Variability for item
               - 'su': supported for users
               - 'si': supported for items
               - 'knn': supported for KNN
               - 'nc': new metric in our articles
        :type confidence_measure: str, default 'vi'

        :param number_sample: Number of new samples (unlabeled samples) per user, which should be labeled   选择的未评分数据数目
        :type number_sample: int, default 10

        :param m: Number of most confident examples to select in each interaction           每次选择最置信的例子数量
        :type m: int, default None

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param ensemble_method: Active ensemble method (ECoRec), combining the n other recommenders outputs 集成方式
        :type ensemble_method: bool, default False

        """

        self.train_file = train_file
        self.test_file = test_file
        self.train_set = ReadFile(train_file, sep=sep).read()
        self.test_set = ReadFile(test_file, sep=sep).read()
        self.recommenders = recommenders
        self.confidence_measure = confidence_measure
        self.number_sample = number_sample
        self.measure_file = None
        self.m = m

        self.sep = sep
        self.ensemble_method = ensemble_method

        # internal vars
        self.number_of_recommenders = len(recommenders)
        self.unlabeled_set = None
        self.unlabeled_data = dict()
        self.labeled_files = dict()
        self.new_file = dict()                      # 添加新文件 每次迭代的预测+ 原有数据
        self.unlabeled_files = dict()
        self.ensemble_file = None

        self.empty = False
        self.recommenders_predictions = None
        self.recommenders_confident = None

        self.rec_conf = dict()
        self.l_test = list()
        self.neighbors_list = {}
        np.random.seed(123)

    def create_unlabeled_set(self):                         # 构造未评分数据集
        """
        Create a pool U' for unlabeled set U (Create unlabeled_1.dat, unlabeled_2.dat ... unlabeled_N.dat)
        """
        unlabeled_data = list()

        for user in self.train_set['users']:
            sample = list(set(self.train_set['items']) - set(self.train_set['feedback'].get(user, [])))
            sub_sample = list()
            for item in sample:
                sub_sample.append((user, item))

            unlabeled_data += sub_sample[: self.number_sample]
        np.random.shuffle(unlabeled_data)
        # Calculate the number of M confident examples based on unlabeled sample
        if self.m is None:
            self.m = int(len(unlabeled_data) * .25)

        unlabeled_data = sorted(unlabeled_data, key=lambda x: (x[0], x[1]))
        self.unlabeled_set = unlabeled_data.copy()

    def create_initial_files(self):                                 # 构造初始文件
        """
        Create labeled and unlabeled files for N recommenders
        """
        for r in self.recommenders:
            labeled_file = os.path.dirname(self.train_file) + '/labeled_set_' + str(r) + '.dat'
            self.labeled_files.setdefault(r, labeled_file)
            unlabeled_file = os.path.dirname(self.train_file) + '/unlabeled_set_' + str(r) + '.dat'
            self.unlabeled_files.setdefault(r, unlabeled_file)
            new_file = os.path.dirname(self.train_file) + '/new_file' + str(r) + '.dat'
            self.new_file.setdefault(r, new_file)

            self.write_with_dict(labeled_file, self.train_set['feedback'])
            self.write_file(self.unlabeled_set, unlabeled_file, score=False)
            self.unlabeled_data.setdefault(r, self.unlabeled_set)

        if self.ensemble_method:
            self.ensemble_file = os.path.dirname(self.train_file) + '/ensemble_set.dat'
            self.write_with_dict(self.ensemble_file, self.train_set['feedback'])

    def train_model(self):                                  # 协同训练部分，迭代直至unlablled集合为空
        """
        Train the model in a co-training process
        """

        epoch = 0
        while not self.empty:
            print("Epoch:: ", epoch)
            self.recommenders_predictions = dict()
            self.recommenders_confident = dict()

            if not self.run_recommenders_parallel():
                break

            self.learn_confident_parallel()
            self.update_data()

            epoch += 1

    # Proposed Confidence           计算置信度
    def pc(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        bui = ComputeBui(label_data).execute()

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]

            nu = len(label_data['items_seen_by_user'].get(user, []))
            ni = len(label_data['users_viewed_item'].get(item, []))

            # compute bui and error
            try:
                den = np.fabs(bui[user][item] - feedback)
            except KeyError:
                den = np.fabs(label_data['mean_value'] - feedback)

            if den == 0:
                den = 0.001

            # compute confidence
            c = (nu * ni) * (1 / den)

            if c != 0:
                confident.append((user, item, feedback, c))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    # 新增置信度 K邻近置信度
    def knn(self, r):
        # 将预测的评分加入-计算相似度
        for r in self.recommenders:
            label_data = ReadFile(self.labeled_files[r]).read()
            self.write_with_dict(self.new_file[r], label_data['feedback'])
            self.update_file(self.recommenders_predictions[r], self.new_file[r])
        rec = UserKNN(self.new_file[r], as_similar_first=True)   # 计算相似度需要加入预测的评分
        rec.read_files()
        rec.init_model()
        confident = list()
        num1 = 0
        # knn_dict[r] = rec.getknn(self.recommenders_predictions[r])
        print(len((self.recommenders_predictions[r])))
        # 给每个预测添加置信度
        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]
            # 得到置信度
            neighbor, abc = rec.getknn(user, item)                      # 获得邻近用户评分
            if abc:
                c = 1                                                   # 没有邻近集的项
            else:
                vi = len(neighbor)
                mean = np.mean(neighbor)  # 均值
                for u in neighbor:
                    num1 += np.abs(u - mean)
                c = np.float32(vi / num1)  # 结果

            if c != 0:
                # print(c.get(user))
                confident.append((user, item, feedback, c))
        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    # Variability for item
    def vi(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        ci = {}

        for item in label_data['items']:
            list_rating = []
            for user in label_data['users_viewed_item'][item]:
                list_rating.append(label_data['feedback'][user][item])
            ci[item] = np.std(list_rating)

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]
            confident.append((user, item, feedback, ci.get(item, 0)))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    # Support for user
    def su(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        ci = {}

        for user in label_data['users']:
            ci[user] = len(label_data['items_seen_by_user'][user])

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]
            confident.append((user, item, feedback, ci.get(user, 0)))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    # Support for item
    def si(self, r):
        confident = list()
        label_data = ReadFile(self.labeled_files[r]).read()
        ci = {}

        for item in label_data['items']:
            ci[item] = len(label_data['users_viewed_item'][item])

        for triple in self.recommenders_predictions[r]:
            user, item, feedback = triple[0], triple[1], triple[2]
            confident.append((user, item, feedback, ci.get(item, 0)))

        confident = sorted(confident, key=lambda y: -y[3])
        complete_confident = confident[:self.m]
        confident = [(x[0], x[1], x[2]) for x in confident]

        return complete_confident, confident[:self.m]

    def learn_confident_parallel(self):                         # 并行学习置信度并实时修改
        pool = Pool()

        method = getattr(self, self.confidence_measure)
        result = pool.map(method, self.recommenders)
        pool.close()
        pool.join()

        for n, r in enumerate(self.recommenders):
            self.rec_conf.setdefault(r, result[n][0])
            self.recommenders_confident.setdefault(r, result[n][1])

    def update_data(self):                                      # 修改数据改变未标记和已标记数据集
        np.random.seed(0)
        n_rec = list(self.recommenders).copy()
        cond = True
        while cond:
            shuffle(n_rec)
            if not [i for i, j in zip(n_rec, self.recommenders) if i == j]:
                cond = False

        for n, r in enumerate(self.recommenders):               # 修改每个基础推荐器的各自的数据
            if self.recommenders_confident[r]:
                self.update_file(self.recommenders_confident[n_rec[n]], self.labeled_files[r])

                rec_conf = [(x[0], x[1]) for x in self.recommenders_confident[r]]
                self.unlabeled_data[r] = list(set(self.unlabeled_data[r]) - set(rec_conf))          # 删除最置信的前n个
                self.write_file(self.unlabeled_data[r], self.unlabeled_files[r], score=False)
            else:
                self.empty = True

    def ensemble(self):                                         # 集成过程
        ensemble_data = list()
        final_rui = {}
        final_dev = {}
        predictions = []
        for r in self.recommenders:
            for conf_sample in self.rec_conf[r]:
                user, item, rui, conf = conf_sample
                if final_rui.get(user, -1) == -1:
                    final_rui.setdefault(user, {}).update({item: rui})
                    final_dev.setdefault(user, {}).update({item: conf})
                else:
                    if final_rui[user].get(item, -1) == -1:
                        final_rui[user][item] = (final_rui[user].get(item, 0) + rui)
                        final_dev[user][item] = (final_dev[user].get(item, 0) + conf)
                    else:
                        if conf > final_dev[user][item]:
                            final_dev[user][item] = rui

        for user in final_rui:
            for item in final_rui[user]:
                rui = final_rui[user][item]

                if rui > self.train_set['max_value']:
                    rui = self.train_set['max_value']
                elif rui < self.train_set['min_value']:
                    rui = self.train_set['min_value']

                ensemble_data.append((user, item, rui))
                predictions.append((user, item, rui))               # 添加的prediction
        self.update_file(ensemble_data, self.ensemble_file)
        return final_rui                                            # 得到集成预测的字典

    def write_file(self, triples, write_file, score=True):          # 写文件
        with open(write_file, 'w') as infile:
            if score:
                for t in triples:
                    infile.write(str(t[0]) + self.sep + str(t[1]) + self.sep + str(t[2]) + '\n')
            else:
                for t in triples:
                    infile.write(str(t[0]) + self.sep + str(t[1]) + self.sep + '1.0 \n')

    def write_with_dict(self, write_file, dict_data):               # 将字典写入文件
        """
        Method to write using data as dictionary. e.g.: user: {item : score}

        """

        with open(write_file, 'w') as infile:
            for user in dict_data:
                for pair in dict_data[user]:
                    infile.write('%d%s%d%s%f\n' % (user, self.sep, pair, self.sep,
                                                   dict_data[user][pair]))

    def update_file(self, triples, write_file):                     # 修改文件
        with open(write_file, 'a') as infile:
            for t in triples:
                infile.write(str(t[0]) + self.sep + str(t[1]) + self.sep + str(t[2]) + '\n')

    def run_recommenders_parallel(self):                            # 并行运行基础推荐器
        """
            create a method to run in parallel the recommenders during the co-training process

        :return: True ou False, if threshold is reached (False) or not (True)
        """

        flag = True

        pool = Pool()
        result = pool.map(self.prediction, self.recommenders)
        pool.close()
        pool.join()

        for n, r in enumerate(self.recommenders):
            if not result[n][1]:
                flag = False
            else:
                self.recommenders_predictions.setdefault(r, result[n][0])

        return flag

    def prediction(self, r):        # 从此函数中返回所需的预测（基础算法预测以及集成的算法预测）
        """
        1: Item KNN
        2: User KNN
        3: Matrix Factorization
        4: SVD++
        """

        flag = True

        if not self.unlabeled_data[r]:
            flag = False

            return [], flag

        else:
            if r == 1:
                rec = ItemKNN(self.labeled_files[r], self.unlabeled_files[r], as_similar_first=True)
                rec.read_files()
                rec.init_model()
                rec.train_baselines()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)
            elif r == 2:
                rec = UserKNN(self.labeled_files[r], self.unlabeled_files[r])
                rec.read_files()
                rec.init_model()
                rec.train_baselines()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)
            elif r == 3:
                rec = MatrixFactorization(self.labeled_files[r], self.unlabeled_files[r], random_seed=1, baseline=True)
                rec.read_files()
                rec.init_model()
                rec.fit()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)
            elif r == 4:
                rec = SVDPlusPlus(self.labeled_files[r], self.unlabeled_files[r], random_seed=1)
                rec.read_files()
                rec.fit()
                rec.predict()
                self.recommenders_predictions.setdefault(r, rec.predictions)

            else:
                raise NameError('Invalid Recommender!')

            return rec.predictions, flag

    @staticmethod
    def transform_dict(list_to_transform):
        new_dict = dict()
        for t in list_to_transform:
            user, item, feedback = t[0], t[1], t[2]
            new_dict.setdefault(user, {}).update({item: feedback})
        return new_dict

    def del_unlabeled_files(self):                              # 删除未标记数据数据
        for f in self.unlabeled_files:
            os.remove(self.unlabeled_files[f])

    def compute(self):

        # print("[Case Recommender: Item Recommendation > %s]\n" % self.recommender_name)
        self.create_unlabeled_set()
        self.create_initial_files()

        self.train_model()

        if self.ensemble_method:
            self.ensemble()

        self.del_unlabeled_files()
        # self.recommender()                            # 未完成


