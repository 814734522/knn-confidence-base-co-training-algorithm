# coding=utf-8

from caserec.recommenders.rating_prediction.userknn import UserKNN
from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.corec import ECoRec


tr_film = '../../../datasets/filmtrust/rating_file.txt'              # 原训练集
te_film = '../../../datasets/filmtrust/folds1/0/test.dat'               # 测试集

label_set1 = '../../../datasets/filmtrust/folds1/0/labeled_set_1.dat'   # 标记数据1
label_set2 = '../../../datasets/filmtrust/folds1/0/labeled_set_2.dat'   # 标记数据2
ensemble = '../../../datasets/filmtrust/folds1/0/ensemble_set.dat'      # 集成数据


if __name__ == '__main__':

    '''置信度设为'pc'、number_sample表示为每个用户预测的对象数目实验中用（20，30，40）'''

    ECoRec(tr_film, te_film, number_sample=20, confidence_measure='knn', ensemble_method=True).compute()       # 运行划分三个文件（每个用户预测20个对象评分）

    # UserKNN(tr_film, te_film, as_similar_first=True, k_neighbors=60).compute()

    # UserKNN(label_set1, te_film, as_similar_first=True, k_neighbors=60).compute()

    # UserKNN(label_set2, te_film, as_similar_first=True, k_neighbors=60).compute()

    # UserKNN(ensemble, te_film, as_similar_first=True, k_neighbors=60).compute()

    # ItemKNN(tr_film, te_film, as_similar_first=True).compute()

    ItemKNN(label_set1, te_film, as_similar_first=True).compute()                       # Item_KNN以Labled_set_1为训练集的性能结果

    ItemKNN(label_set2, te_film, as_similar_first=True).compute()                       # Item_KNN以Labled_set_2为训练集的性能结果

    ItemKNN(ensemble, te_film, as_similar_first=True).compute()                         # Item_KNN以ensemble_set为训练集的性能结果

    # ECoRec(tr_film, te_film, number_sample=30, confidence_measure='ec', ensemble_method=True).compute()

    # ItemKNN(label_set1, te_film, as_similar_first=True).compute()

    # ItemKNN(label_set2, te_film, as_similar_first=True).compute()

    # ItemKNN(ensemble, te_film, as_similar_first=True).compute()

    # ECoRec(tr_film, te_film, number_sample=40, confidence_measure='ec', ensemble_method=True).compute()

    # ItemKNN(label_set1, te_film, as_similar_first=True).compute()

    # ItemKNN(label_set2, te_film, as_similar_first=True).compute()

    # ItemKNN(ensemble, te_film, as_similar_first=True).compute()