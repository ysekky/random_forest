# -*- coding:utf-8 -*-

__author__ = 'ysekky'

from collections import defaultdict
import math


class Node(object):
    """
    分類用のNodeのクラス
    """
    left_node = None
    right_node = None

    def __init__(self, feature_name, samples, threshold):
        """

        :param feature_name: string
        :param samples: Samples
        :param threshold: float
        :return:
        """
        self.feature_name = feature_name
        self.samples = samples
        self.threshold = threshold
        self.left_samples, self.right_samples = self.divide_sample()
        self.information_gain = self.calc_information_gain()


    def divide_sample(self):
        left_samples = Samples()
        right_samples = Samples()
        for sample in self.samples:
            if self.threshold <= sample.get(self.feature_name):
                left_samples.append(sample)
            else:
                right_samples.append(sample)
        return left_samples, right_samples


    def calc_information_gain(self):
        left_probability = float(len(self.left_samples))/len(self.samples)
        right_probability = float(len(self.right_samples))/len(self.samples)
        information_gain = (
            self.samples.entropy
            - left_probability * self.left_samples.entropy
            - right_probability * self.right_samples.entropy
        )
        return information_gain

    def predict(self, sample):
        if self.threshold <= sample.get(self.feature_name):
            if self.left_node:
                return self.left_node.predict(sample)
            return self.left_samples.label_probability
        if self.right_node:
            return self.right_node.predict(sample)
        return self.right_samples.label_probability


class Sample(object):
    """
    データ入力
    """

    def __init__(self, name, features, label):
        self.name = name
        self.features = features
        self.label = label

    def get(self, feature_name):
        return self.features[feature_name]


class Samples(list):
    """
    Sampleの集合．ラベルの確率とエントロピーを属性として追加
    """

    _entropy = None
    _label_probability = None

    @property
    def entropy(self):
        if self._entropy:
            return self._entropy
        self._entropy = self.__calc_entropy()
        return self._entropy

    @property
    def label_probability(self):
        if self._label_probability:
            return self._label_probability
        self._label_probability = self.__calc_label_probability()
        return self._label_probability

    def __calc_label_probability(self):
        label_counts = defaultdict(float)
        for sample in self:
            label_counts[sample.label] += 1.0
        label_probability = {}
        for label, count in label_counts.iteritems():
            label_probability[label] =  count/len(self)
        return label_probability

    def __calc_entropy(self):
        """
        サンプル集合から目的ラベルのエントロピーを計算する
        :return:
        """
        entropy = .0
        for label, probability in self.label_probability.iteritems():
            entropy -= probability * math.log(probability, 2)
        return entropy



def run(samples):
    node_candidate = create_node_candidates(samples)
    node = select_node(samples, node_candidate)
    if not node:
        return None
    if node.left_samples.entropy != 0:
        node.left_node = run(node.left_samples)
    if node.right_samples.entropy != 0:
        node.right_node = run(node.right_samples)
    return node


def select_node(samples, candidates):
    """
    sampleのノードを作って一番いいのを得る
    :param samples:
    :return:
    """
    max_ig = None
    best_node = None
    for (feature_name, value), used in candidates.iteritems():
        if used:
            #使われた条件のノードは生成しない
            continue
        node = Node(feature_name, samples, value)
        if not max_ig or max_ig < node.information_gain:
            max_ig = node.information_gain
            best_node = node
    #使ったやつは条件からはずす
    candidates[best_node.feature_name, best_node.threshold] = True
    return best_node


def create_node_candidates(samples):
    """
    ノードの条件リストを生成する
    :param samples:
    :return:
    """
    features = defaultdict(list)
    candidates = {}
    for sample in samples:
        for feature_name, value in sample.features.iteritems():
            features[feature_name].append(value)
    for feature_name, values in features.iteritems():
        values = list(set(values))
        values.sort()
        for value in values:
            candidates[(feature_name, value)] = False
    return candidates


def create_sample():
    samples = Samples()
    for line in open("sample.csv"):
        name, q1, q2, q3, label = line[:-1].split(',')
        features = {'q1':int(q1), 'q2':int(q2), 'q3':int(q3)}
        samples.append(Sample(name=name, features=features, label=label.strip()))
    return samples

from sklearn import datasets


def create_iris_sample():
    iris = datasets.load_iris()
    train_samples = Samples()
    test_sample = Samples()
    for index, data in enumerate(iris.data):
        features = {'q0': data[0], 'q1': data[1], 'q2': data[2], 'q3': data[3]}
        if index % 2 == 0:
            train_samples.append(Sample(name=index, features=features, label=iris.target_names[iris.target[index]]))
        else:
            test_sample.append(Sample(name=index, features=features, label=iris.target_names[iris.target[index]]))
    return train_samples, test_sample



def run_by_iris():
    train_samples, test_samples = create_iris_sample()
    node = run(train_samples)
    correct_count = 0
    for sample in test_samples:
        result = node.predict(sample)
        if result.get(sample.label, 0.0) == 1.0:
            correct_count += 1.0
    print correct_count/len(test_samples)
    return node