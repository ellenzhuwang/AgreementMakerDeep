# coding:utf-8
import numpy as np
import tensorflow as tf
import json


class Trans:
    def get_config(self):
        return self.config

    def input_def(self):
        config = self.config
        self.batch_n = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.batch_m = tf.placeholder(tf.int64, [config.batch_seq_size])
        self.postive_n = tf.transpose(tf.reshape(self.batch_n[0:config.batchsize], [1, -1]), [1, 0])
        self.postive_m = tf.transpose(tf.reshape(self.batch_m[0:config.batchsize], [1, -1]), [1, 0])
        self.relation_n = tf.transpose(tf.reshape(self.batch_n[0:config.batchsize], [1, -1]), [1, 0])
        self.negative_n = tf.transpose(tf.reshape(self.batch_n[config.batchsize:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
        self.negative_m = tf.transpose(tf.reshape(self.batch_m[config.batchsize:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
        self.relation_m = tf.transpose(tf.reshape(self.batch_m[config.batchsize:config.batch_seq_size], [config.negative_ent, -1]), perm=[1, 0])
        self.parameter_lists = []

    def __init__(self, config):
        self.config = config

        with tf.name_scope("input"):
            self.input_def()

        with tf.name_scope("embedding" + "relation"):
            self.embedding_def()

        with tf.name_scope("instance"):
            self.pre_instance()

        with tf.name_scope("pro_loss"):
            self.projection_loss()

    def embedding_def(self):
        config = self.get_config()
        self.source_ent_embeddings = tf.get_variable(name="source_ent_embeddings", shape=[config.sourceenttotal, config.dimension], initializer=tf.contrib.layers.xavier_initializer(uniform=False))                                                 
        self.target_ent_embeddings = tf.get_variable(name="target_ent_embeddings", shape=[config.targetrelationtotal, config.dimension], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.source_rel_embeddings = tf.get_variable(name="source_rel_embeddings", shape=[config.sourceenttotal, config.dimension], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.target_rel_embeddings = tf.get_variable(name="target_rel_embeddings", shape=[config.targetrelationtotal, config.dimension], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.parameter_lists = {"source_ent_embeddings": self.source_ent_embeddings,"target_ent_embeddings": self.target_ent_embeddings}

    def get_positive_instance(self, in_batch=True):
        return [self.postive_n, self.postive_m, self.relation_n]

    def get_negative_instance(self, in_batch=True):
        return [self.negative_n, self.negative_m, self.relation_m]

    def pre_instance(self):
        self.phs, self.pts, self.prs = self.get_positive_instance(in_batch=True)
        self.nhs, self.nts, self.nrs = self.get_negative_instance(in_batch=True)

    def projection_loss(self):
        config = self.get_config()
        phs = tf.reshape(tf.nn.embedding_lookup(self.source_ent_embeddings, self.phs), [-1, config.dimension, 1])
        prs = tf.reshape(tf.nn.embedding_lookup(self.source_rel_embeddings, self.prs), [-1, config.dimension, 1])
        pts = tf.reshape(tf.nn.embedding_lookup(self.target_ent_embeddings, self.pts), [-1, config.dimension, 1])
        nhs = tf.reshape(tf.nn.embedding_lookup(self.source_ent_embeddings, self.nhs), [-1, config.dimension, 1])
        nrs = tf.reshape(tf.nn.embedding_lookup(self.target_rel_embeddings, self.nrs), [-1, config.dimension, 1])
        nts = tf.reshape(tf.nn.embedding_lookup(self.target_ent_embeddings, self.nts), [-1, config.dimension, 1])
        p_score = tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1)
        n_score = tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1)
        pos_loss = tf.reduce_sum(tf.maximum(p_score - tf.constant(0.01), 0))
        neg_loss = 0.2 * tf.reduce_sum((tf.maximum(tf.constant(2.0) - n_score, 0)))
        self.pro_loss = pos_loss + neg_loss

    def generate_optimizer(loss):
        opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("relation")]
        optimizer = tf.train.AdagradOptimizer(self.lr).minimize(loss, var_list=opt_vars)
        return optimizer
