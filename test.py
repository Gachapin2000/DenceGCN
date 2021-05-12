import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
import torch

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/planetoid/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/planetoid/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = np.random.choice(2708, int(2708*0.2), replace=False)
    without_test=np.array([i for i in range(2708) if i not in idx_test ])
    idx_train = without_test[np.random.choice(np.arange(len(without_test)),int(2708*0.6), replace=False)]
    idx_val = np.array([i for i in range(2708) if i not in idx_test if i not in idx_train])

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, np.array(y_train), np.array(y_val), np.array(y_test), np.array(train_mask), np.array(val_mask), np.array(test_mask), graph


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)



class JKNet:
    def __init__(self, layer, n_nodes, f_dimension, nb_classes, initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
        self.X = tf.placeholder('float32',shape=(n_nodes*f_dimension))
        self.y = tf.placeholder('float32', shape=(n_nodes, nb_classes))
        self.mask=tf.placeholder('float32')
        self.nodes=n_nodes
        self.output_dim = nb_classes
        self.layer = layer
        self.is_train=tf.placeholder(tf.bool)
        self.adj=tf.placeholder('float32',shape=(n_nodes, n_nodes))
        self.dropout=tf.placeholder(tf.float32)
        self.init=initializer
        
    def loss(self, lr, active=tf.nn.relu, optim=tf.train.AdamOptimizer, JK_Model='concat'):
        self.L=tf.identity(tf.reshape(self.X, [self.nodes, -1]))
        _layers= []
        for i in range(len(self.layer)):
            self.L=self.graph_conv(self.L, self.adj, self.layer[i], activation=active)
            self.L= tf.nn.dropout(self.L, self.dropout)
            _layers.append(self.L)
            
        if JK_Model=='concat':
            hypo= tf.concat(_layers, axis=1)

            
        elif JK_Model=='MaxPooling':
            hypo= tf.stack(_layers, axis=0)   
            hypo = tf.reduce_max(hypo, axis=0)

            
        elif JK_Model=='Bi-LSTM Attention':
            self.Attention=self.Bi_directional_lstm_attention(_layers)
            hypo=0
            for i in range(len(_layers)):
                hypo+=self.Attention[i]*_layers[i]

            
        elif JK_Model=='Only_GCN':
            hypo=self.L

        elif JK_Model=='Only_GAT':
            self.hidden_layer=tf.identity(self.L)
            for i, hidden_layer in enumerate(self.layer):
                self.hidden_layer = tf.layers.dense(self.hidden_layer, hidden_layer, kernel_initializer=self.init)
                self.coeffs = self.attn_coeffs_DGL(self.hidden_layer, self.adj)
                self.hidden_layer = tf.matmul(self.coeffs, self.hidden_layer) # [batch_size, num_nodes, num_nodes]  
                self.hidden_layer = active(self.hidden_layer)
                self.hidden_layer = tf.nn.dropout(self.hidden_layer, self.dropout)
            hypo=self.hidden_layer

        else:
            raise ValueError("You should assign JK_model")
        
        
        
        hypo=tf.matmul(self.adj, hypo)
        self.influence=self.Influence_x_y(11, hypo)
        self.hypothesis = tf.layers.dense(hypo, self.output_dim, activation=tf.nn.softmax, kernel_initializer=self.init)
        
        var   = tf.trainable_variables()     
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var if 'bias' not in v.name ]) *  0.0005  
        
        self.cost=tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.hypothesis)+lossL2
        mask = tf.cast(self.mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        

        self.cost *= mask       
        self.cost=tf.reduce_mean(self.cost)
        
        self.optimizer = optim(learning_rate=lr)
        self.trains = self.optimizer.minimize(self.cost)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def graph_conv(self, _X, _A, output_dim, activation=tf.nn.relu):
        output = tf.layers.dense(_X, units=output_dim, kernel_initializer=self.init)
        output = tf.matmul(_A, output)
        output = activation(output)
        return output
    
    def Bi_directional_lstm_attention(self, Layer):
        l=[]
        for layer in Layer: # each layer (n, d) * l
            l.append(tf.reshape(layer, [1,-1])) # -> (1, nd) * l
        X=tf.stack(l, axis=1) (1, ndl)
        
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = self.layer[0], state_is_tuple = True)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout)
        
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units = self.layer[0], state_is_tuple = True)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout)
        
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, X, dtype = tf.float32)

        outputs_fw = tf.reshape(outputs[0], [len(Layer),self.layer[0]])
        outputs_bw = tf.reshape(outputs[1], [len(Layer),self.layer[0]])
        outputs_concat = tf.concat([outputs_fw, outputs_bw], axis=1)
        concatenated=tf.layers.dense(outputs_concat, units=1, use_bias=True, kernel_initializer=self.init)
        s=tf.nn.softmax(tf.reshape(concatenated,[-1]))
        return s
                    
    def Influence_x_y(self, Node_x, hypo):
        abs_grad=tf.math.abs(tf.gradients(tf.reshape(hypo[Node_x], [-1]), self.X)[0])
        abs_grad=tf.reshape(abs_grad,[self.nodes,-1])
        jacobian=tf.reduce_sum(abs_grad,axis=1)
        Influence_of_Y_ON_X=jacobian/tf.reduce_sum(jacobian)
        return Influence_of_Y_ON_X
        
    def attn_coeffs_DGL(self, H, A):
        H=tf.matmul(A,H)
        logits = tf.layers.dense(H, self.nodes,activation=tf.nn.leaky_relu)
        zero_vec = -9e15*tf.ones_like(logits)
        attention = tf.where(A > 0, logits, zero_vec)
        coefs = tf.nn.softmax(attention, axis=1)
        return coefs
    
    def accuracy(self, X, Y, mask, adj):
        """Accuracy with masking."""
        preds, influence=self.sess.run([self.hypothesis, self.influence], feed_dict={self.X:X, self.y:Y, self.adj:adj,
                                                                              self.mask:mask, self.dropout:1})
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return self.sess.run(tf.reduce_mean(accuracy_all)), influence
    
        
    
    def train(self, batch_xs, batch_ys, mask, avg_cost, adj):
        _, c = self.sess.run([self.trains, self.cost], feed_dict={self.X: batch_xs, self.y: batch_ys,  self.adj:adj,
                                                                  self.mask:mask, self.dropout:0.5})
        avg_cost += np.mean(c)
        return avg_cost



tf.reset_default_graph()
model_cora=JKNet([32,32,32,32], 2708, 1433, 7)
model_cora.loss(5e-3, active=tf.nn.relu, JK_Model='Bi-LSTM Attention')
for epoch in range(5):
    avg_cost=0
    for i in range(40):
        batch_xs, batch_ys, adj=features_cora.reshape(-1),  y_train_cora, adj_norm_cora
        avg_cost=model_cora.train(batch_xs, batch_ys, train_mask_cora, avg_cost/100, adj)
    train_acc, _=model_cora.accuracy(batch_xs, batch_ys, train_mask_cora, adj)
    print('Epoch:','%04d' %(epoch+1),'train_acc: ',train_acc)
test_acc, _=model_cora.accuracy(batch_xs, y_test_cora, test_mask_cora, adj)
print('Test_acc: ',test_acc)