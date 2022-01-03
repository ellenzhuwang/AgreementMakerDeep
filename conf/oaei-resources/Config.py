# coding:utf-8
import numpy as np
import tensorflow as tf
import json
from tqdm import tqdm
import random
import timeit
import os
import itertools


class Batch(object):

    def get_batch(self):
        n_triple = len(self.triple_train)
        rand_idx = np.random.permutation(n_triple)
        start = 0
        batchsize = int(n_triple / self.nbatches)
        while start < batchsize * self.nbatches:
            start_t = timeit.default_timer()
            end = min(start + batchsize, n_triple)
            size = end - start
            train_triple_positive = list([self.triple_train[x] for x in rand_idx[start:end]])
            train_triple_negative = []
            for t in train_triple_positive:
                random_num = np.random.random()
                source_entity_id_list = list(range(self.sourceentitytotal))
                target_entity_id_list = list(range(self.targetentitytotal))
                source_entity_id_list.remove(t[0])
                target_entity_id_list.remove(t[1])
                if str(t[0]) in self.source_constrain_dict['dij'].keys():
                    dis_samp_source = self.source_constrain_dict['dij'][str(t[0])]
                else: dis_samp_source = []
                if str(t[1]) in self.target_constrain_dict['dij'].keys():
                    dis_samp_target = self.target_constrain_dict['dij'][str(t[1])]
                else: dis_samp_target = []
                if self.negative_ent <= len(dis_samp_source):
                    replace_source_entity_id_list = random.sample(dis_samp_source, self.negative_ent)
                else:
                    try:
                        for ntri in self.source_constrain_dict['sbc'][str(t[0])] + dis_samp_source:
                            if ntri in source_entity_id_list: 
                                source_entity_id_list.remove(ntri)
                            else:
                                pass
                    except KeyError:
                        pass
                    tem = random.sample(source_entity_id_list, self.negative_ent - len(dis_samp_source))
                    replace_source_entity_id_list = dis_samp_source + tem
                if self.negative_ent <= len(dis_samp_target):
                    replace_target_entity_id_list = random.sample(dis_samp_target, self.negative_ent)
                else:
                    try:
                        for targettri in self.target_constrain_dict['sbc'][str(t[1])] + dis_samp_target:
                            if targettri in target_entity_id_list: 
                                target_entity_id_list.remove(targettri)
                            else:
                                pass
                    except KeyError:
                        pass
                    tem = random.sample(target_entity_id_list, self.negative_ent - len(dis_samp_target))
                    replace_target_entity_id_list = dis_samp_target + tem
                if self.negative_sampling == 'unif':
                    replace_source_probability = 0.5
                else:
                    pass
                if self.modelname == "trans":
                    for mt in replace_target_entity_id_list:
                        train_triple_negative.append((t[0], mt))
                else:
                    pass
                self.p_positive_batch_n = list([x[0] for x in train_triple_positive])
                self.p_positive_batch_m = list([x[1] for x in train_triple_positive])
                self.p_negative_batch_n = list([triple[0] for triple in train_triple_negative])
                self.p_negative_batch_m = list([triple[1] for triple in train_triple_negative])
                self.p_batch_n = self.p_positive_batch_n + self.p_negative_batch_n
                self.p_batch_m = self.p_positive_batch_m + self.p_negative_batch_m
            start = end
            prepare_t = timeit.default_timer() - start_t
            if self.modelname == "trans":
                yield self.p_batch_n, self.p_batch_m, prepare_t
            else:
                pass

    def __init__(self):
        self.negative_sampling = 'unif'
        self.triple_train = []
        self.sourceentitytotal = 0
        self.targetentitytotal = 0
        self.sourcerelationtotal = 0
        self.targetrelationtotal = 0
        self.sourceent2id = {}
        self.targetent2id = {}
        self.tripletotal = 0
        self.constrain_source_tripletotal = 0
        self.constrain_target_tripletotal = 0
        self.in_path = ""
        self.nbatches = 0
        self.negative_ent = 1
        self.modelname = None

    def readData(self):

        with open(os.path.join(self.in_path, 'ent_ids_source.txt')) as f:
            self.sourceent2id = {line.strip().split('\t')[1]: int(line.strip().split('\t')[0]) for line in f.readlines()}
        with open(os.path.join(self.in_path, 'ent_ids_target.txt')) as f:
            self.targetent2id = {line.strip().split('\t')[1]: int(line.strip().split('\t')[0]) for line in f.readlines()}
        with open(os.path.join(self.in_path, 'neg_constrain_source.json')) as f:
            self.source_constrain_dict = json.load(f)
        with open(os.path.join(self.in_path, 'neg_constrain_target.json')) as f:
            self.target_constrain_dict = json.load(f)
        self.triple_train = self.readTriple('train.txt')
        self.sourceentitytotal = len(self.sourceent2id)
        self.targetentitytotal = len(self.targetent2id)
        self.sourcerelationtotal = len(self.sourceent2id)
        self.targetrelationtotal = len(self.targetent2id)
        self.tripletotal = len(self.triple_train)
       

    def readTriple(self, filename):
        triple_list = []
        train_list = []
        with open(os.path.join(self.in_path, filename)) as f:
            for line in f.readlines():
                train_list = line.strip().split('\t')
                triple_list.append((int(train_list[0]), int(train_list[1])))

        return triple_list

    def setBatches(self, nbatches):
        self.nbatches = nbatches

    def inPath(self, path):
        self.in_path = path

    def negRate(self, rate):
        self.negative_ent = rate

    def negativeSampling(self, negative_sampling):
        self.negative_sampling = negative_sampling

    def model_name(self, name):
        self.modelname = name

class Config(Batch):

    def __init__(self):
        Batch.__init__(self)
        self.out_path = None
        self.train_times = 0
        self.alpha = 0.001
        self.log_on = 1
        self.dimension = 100
        self.exportName = None
        self.importName = None
        self.export_steps = 0
        self.opt_method = "SGD"
        self.optimizer = None


    def init(self):
        if self.in_path != None:
            self.readData()
            self.sourceenttotal = self.sourceentitytotal
            self.targetenttotal = self.targetentitytotal
            self.sourcereltotal = self.sourcerelationtotal
            self.targetreltotal = self.targetrelationtotal
            self.batchsize = int(self.tripletotal / self.nbatches)
            self.batch_seq_size = self.batchsize * (1 + self.negative_ent)
            self.batch_n = np.zeros(self.batchsize * (1 + self.negative_ent), dtype=np.int64)
            self.batch_m = np.zeros(self.batchsize * (1 + self.negative_ent), dtype=np.int64)

    def optimizer(self, optimizer):
        self.optimizer = optimizer

    def optMethod(self, method):
        self.opt_method = method

    def learningRate(self, alpha):
        self.alpha = alpha

    def vecFiles(self, path):
        self.out_path = path

    def entDimension(self, dim):
        self.dimension = dim

    def trainTimes(self, times):
        self.train_times = times

    def exportFiles(self, path, steps=0):
        self.exportName = path
        self.export_steps = steps

    def saveTF(self):
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, self.exportName)

    def parameters_name(self, var_name):
        with self.graph.as_default():
            with self.sess.as_default():
                if var_name in self.trainModel.parameter_lists:
                    return self.sess.run(self.trainModel.parameter_lists[var_name])
                else:
                    return None

    def get_parameters(self, mode="numpy"):
        res = {}
        lists = self.trainModel.parameter_lists
        for var_name in lists:
            if mode == "numpy":
                res[var_name] = self.parameters_name(var_name)
            else:
                res[var_name] = self.parameters_name(var_name).tolist()
        return res

    def save_parameters(self, path=None):
        if path == None:
            path = self.out_path
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def model(self, model):
        self.model = model
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                initializer = tf.contrib.layers.xavier_initializer(uniform=True)
                with tf.variable_scope("model", reuse=None, initializer=initializer):
                    self.trainModel = self.model(config=self)
                    self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
                    grads_and_vars = self.optimizer.compute_gradients(self.trainModel.pro_loss)
                    self.train_op = self.optimizer.apply_gradients(grads_and_vars)
                self.saver = tf.train.Saver()
                self.sess.run(tf.initialize_all_variables())

    def train(self, batch_h, batch_t):
        feed_dict = {
            self.trainModel.batch_n: batch_h,
            self.trainModel.batch_m: batch_t}
        _,pro_loss = self.sess.run([self.train_op, self.trainModel.pro_loss], feed_dict)
        return pro_loss

    def run(self):
        with self.graph.as_default():
            with self.sess.as_default():
                for times in tqdm(range(self.train_times)):
                    pro_res = 0.0
                    if self.modelname == 'trans':
                        for bn, bm, _ in self.get_batch():
                            pro = self.train(bn, bm)
                            pro_res += pro
                    else:
                        pass
                #self.saveTF()
                self.save_parameters(self.out_path)
