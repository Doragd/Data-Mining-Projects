# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class KdTree(object):

    # 获取鸢尾花数据
    def get_iris_data(self):
        iris = load_iris()
        iris_data = iris.data
        iris_target = iris.target

        return iris_data, iris_target

    def run(self):
        # 数据准备
        iris_data, iris_target = self.get_iris_data()
        # 训练集/测试集划分
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)
        # 数据标准化
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)
        # 创建分类器
        kd_tree = self.evaluate()
        kd_tree.fit(x_train, y_train) # 将测试集送入算法
        y_predict = kd_tree.predict(x_test) # 获取预测结果
        # 预测结果展示
        labels = ["山鸢尾","虹膜锦葵","变色鸢尾"]
        for i in range(len(y_predict)):
            print("第%d次测试:真实值:%s\t预测值:%s"%((i+1),labels[y_predict[i]],labels[y_test[i]]))
        print("准确率：",kd_tree.score(x_test, y_test))
    def evaluate(self):
        iris_data, iris_target = self.get_iris_data()
        # 训练集/测试集划分
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)
        # 生成knn估计器
        kd_tree = KNeighborsClassifier(algorithm='kd_tree')
        # 构造超参数值
        params = {"n_neighbors":range(1,11)}
        # 进行网格搜索
        gridCv = GridSearchCV(kd_tree, param_grid=params, cv=5)
        gridCv.fit(x_train,y_train) # 输入训练数据
        # 预测准确率
        print("准确率：",gridCv.score(x_test, y_test))
        print("交叉验证中最好的结果：",gridCv.best_score_)
        print("最好的模型：", gridCv.best_estimator_)
        return gridCv.best_estimator_
    
class KNN(object):

    # 获取鸢尾花数据
    def get_iris_data(self):
        iris = load_iris()
        iris_data = iris.data
        iris_target = iris.target

        return iris_data, iris_target

    def run(self):
        # 数据准备
        iris_data, iris_target = self.get_iris_data()
        # 训练集/测试集划分
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)
        # 数据标准化
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)
        # 创建分类器
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train) # 将测试集送入算法
        y_predict = knn.predict(x_test) # 获取预测结果
        # 预测结果展示
        labels = ["山鸢尾","虹膜锦葵","变色鸢尾"]
        for i in range(len(y_predict)):
            print("第%d次测试:真实值:%s\t预测值:%s"%((i+1),labels[y_predict[i]],labels[y_test[i]]))
        print("准确率：",knn.score(x_test, y_test))
    def evaluate(self):
        iris_data, iris_target = self.get_iris_data()
        # 训练集/测试集划分
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)
        # 生成knn估计器
        knn = KNeighborsClassifier()
        # 构造超参数值
        params = {"n_neighbors":range(1,11)}
        # 进行网格搜索
        gridCv = GridSearchCV(knn, param_grid=params, cv=5)
        gridCv.fit(x_train,y_train) # 输入训练数据
        # 预测准确率
        print("准确率：",gridCv.score(x_test, y_test))
        print("交叉验证中最好的结果：",gridCv.best_score_)
        print("最好的模型：", gridCv.best_estimator_)

if __name__ == '__main__':
    kd_tree = KdTree()
    kd_tree.run()