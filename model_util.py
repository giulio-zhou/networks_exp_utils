import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

class ModelWrapper(object):
    def load_model(self):
        pass
    def predict_floats(self, X):
        pass
    def predict_ints(self, X):
        pass
    def save_model(self, paths):
        pass
    def train_model(self, X, y):
        pass

# Define class of simple Tensorflow models.
# These are intended for simple binary classification tasks.
class TensorflowSimpleModel(ModelWrapper):
    def __init__(self, model_fn, model_path=None,
                 input_dim=1024, name='', save_model=True):
        self.model_fn = model_fn
        self.model_path = model_path
        self.model = None
        self.input_dim = input_dim
        self.name = name
        self.save_model = save_model
        # Start session and construct graph.
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.label_tensor = tf.placeholder(tf.int64, shape=[None])
        self.ex_weight_tensor = tf.placeholder(tf.float32, shape=[None])
        self.loss, self.classes, self.probabilities, self.accuracy = model_fn(
            self.input_tensor, self.label_tensor, self.ex_weight_tensor)
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()

    def delete(self):
        tf.reset_default_graph()

    def load_model(self):
        self.saver.restore(self.sess, self.model_path)

    def predict_floats(self, X):
        softmax_outputs = self.sess.run(
            'probabilities:0', feed_dict={self.input_tensor: X})
        return softmax_outputs[:, 1]

    def predict_ints(self, X):
        return self.predict_floats(X)

    def reset(self):
        tf.reset_default_graph()
        self.__init__(self.model_fn, self.model_path, self.input_dim)

    def save_model(self, paths):
        pass

    def train_model(self, X, y, ex_weights=None, batch_size=256, n_epochs=5,
                    optimizer_fn=tf.train.AdamOptimizer, lr=0.001):
        if ex_weights is None:
            ex_weights = np.ones([len(y)])
        optimizer = optimizer_fn(learning_rate=lr)
        train = optimizer.minimize(self.loss)
        num_iter = int(n_epochs * len(y) / batch_size)
        y = y.astype(np.int32)
        # Class balance : try to achieve class balance if possible, else tries
        # to fill a batch with the maximum number of minority samples.
        if np.sum(y) < 0.5 * batch_size: 
            pos_weight = np.sum(y) / float(batch_size)
            class_balance = [1 - pos_weight, pos_weight]
        elif len(y) - np.sum(y) < 0.5 * batch_size:
            neg_weight = (len(y) - np.sum(y)) / float(batch_size)
            class_balance = [neg_weight, 1 - neg_weight]
        else:
            class_balance = [0.5, 0.5]
        # NOTE: Number of classes currently hardcoded to 2.
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        alpha_pos = ex_weights[pos_idx] / np.sum(ex_weights[pos_idx])
        alpha_neg = ex_weights[neg_idx] / np.sum(ex_weights[neg_idx])
        batch_num_pos = int(class_balance[1] * batch_size)
        batch_num_neg = batch_size - batch_num_pos
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        for i in range(num_iter):
            # Create balanced input batches.
            if batch_num_pos == 0:
                idx = np.random.choice(
                    neg_idx, batch_num_neg, replace=False, p=alpha_neg)
                X_feed, y_feed = X[idx], y[idx]
                alpha_feed = ex_weights[idx]
            elif batch_num_neg == 0:
                idx = np.random.choice(
                    pos_idx, batch_num_pos, replace=False, p=alpha_pos)
                X_feed, y_feed = X[idx], y[idx]
                alpha_feed = ex_weights[idx]
            else:
                pos_ex_idx = np.random.choice(
                    pos_idx, batch_num_pos, replace=False, p=alpha_pos)
                neg_ex_idx = np.random.choice(
                    neg_idx, batch_num_neg, replace=False, p=alpha_neg)
                all_idx = np.hstack([neg_ex_idx, pos_ex_idx])
                X_feed, y_feed = X[all_idx], y[all_idx]
            _, loss, accuracy = \
                self.sess.run([train, self.loss, self.accuracy],
                              feed_dict={self.input_tensor: X_feed,
                                         self.label_tensor: y_feed,
                                         self.ex_weight_tensor: np.ones(len(y_feed))})
            if i % 1 == 0:
                print("@{} - loss: {}, accuracy: {}".format(i, loss, accuracy))
        if self.save_model:
            save_path = self.saver.save(self.sess, self.model_path)
            print("Model saved in file: %s" % save_path)

def simple_classifier(n_hidden=[200], activations=[tf.nn.relu]):
    def model_fn(inputs, labels, ex_weights):
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        onehot_labels = tf.reshape(onehot_labels, [-1, 2])
        # Network layers.
        if len(n_hidden) == 0:
            single_logits = tf.layers.dense(inputs=inputs, units=1)
        else:
            hidden = tf.layers.dense(
                inputs=inputs, units=n_hidden[0], activation=activations[0])
            for i in range(1, len(n_hidden)):
                hidden = tf.layers.dense(
                    inputs=hidden, units=n_hidden[i], activation=activations[i])
            single_logits = tf.layers.dense(inputs=hidden, units=1)
        logits = tf.concat([1 - single_logits, single_logits], axis=1)
        # Loss.
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits,
            reduction=tf.losses.Reduction.NONE)
        loss = tf.multiply(loss, ex_weights)
        loss = tf.reduce_mean(loss, name="loss")
        # Outputs.
        classes = tf.argmax(input=logits, axis=1, name="classes")
        probabilities = tf.nn.softmax(logits, name="probabilities")
        accuracy = tf.contrib.metrics.accuracy(
            labels=labels, predictions=classes)
        return loss, classes, probabilities, accuracy
    return model_fn

def get_simple_tf_model_by_name(model_name):
    if model_name == 'simple_classifier':
        model_fn = simple_classifier
    else:
        print("No valid model named %s" % model_name)
        exit(1)
    return model_fn

# Nearest Neighbor Models.
def compute_pairwise_dists(X, Z):
    """
    Inputs are X (N x d) and Z (M x k).
    Computes the pairwise euclidean distances between X and Z, and returns
    an (N x M) distance matrix D.
    Implementation: Computing the matrix of cross-pairwise distances is
    equivalent to X**2 - 2 * X Z_T + Z**2
    """
    num_X = tf.shape(X)[0]
    num_Z = tf.shape(Z)[0]
    X_squared_norm = tf.square(tf.norm(X, axis=1))
    Z_squared_norm = tf.square(tf.norm(Z, axis=1))
    cross_terms = tf.matmul(X, tf.transpose(Z))
    D = tf.add(-2 * cross_terms, Z_squared_norm)
    D = tf.add(X_squared_norm, tf.transpose(D))
    D = tf.sqrt(tf.transpose(D))
    return D

def run_pairwise_dists(
        sess, X_tensor, Z_tensor, norm_tensor, max_norm_batch_size, X, Z):
    # Compute the pairwise Euclidean norm between X and Z in chunks to ensure
    # that this will fit into GPU memory.
    num_full_norm_batches = len(X) // max_norm_batch_size
    norm_batch_remainder = len(X) % max_norm_batch_size
    norms = []
    for k in range(num_full_norm_batches):
        X_slice = X[k*max_norm_batch_size:(k+1)*max_norm_batch_size]
        norm_slice = sess.run(
            norm_tensor, feed_dict={X_tensor: X_slice, Z_tensor: Z})
        norms.append(norm_slice)
    if norm_batch_remainder > 0:
        X_slice = X[-norm_batch_remainder:]
        norm_slice = sess.run(
            norm_tensor, feed_dict={X_tensor: X_slice, Z_tensor: Z})
        norms.append(norm_slice)
    composite_norm_npy = np.hstack(norms)
    return composite_norm_npy

class SimpleKNNModel(ModelWrapper):
    def __init__(self, k, prediction_thresh,
                 max_norm_batch_size=10000, name=''):
        self.k = k
        self.prediction_thresh = prediction_thresh
        self.name = name
        self.train_data = None
        self.train_labels = None
        self.max_norm_batch_size = max_norm_batch_size
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.training_set_tensor = \
            tf.placeholder(tf.float32, shape=[None, None])
        self.test_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
        self.norm_tensor = compute_pairwise_dists(self.training_set_tensor,
                                                  self.test_set_tensor)
        self.norm_tensor_placeholder = \
            tf.placeholder(tf.float32, shape=[None, None])
        self.top_k_vals_tensor, self.top_k_idx_tensor = \
            tf.nn.top_k(-self.norm_tensor_placeholder, k)
        self.top_k_vals_tensor = -self.top_k_vals_tensor
        self.sess.run(self.init_op)

    def predict_floats(self, X):
        if (self.train_data is None) or (self.train_labels is None):
            raise Exception("Train data and labels have not been instantiated")
        composite_norm_npy = run_pairwise_dists(
            self.sess, self.training_set_tensor, self.test_set_tensor,
            self.norm_tensor, self.max_norm_batch_size, self.train_data, X)
        top_k_idx = self.sess.run(
            self.top_k_idx_tensor,
            feed_dict={self.norm_tensor_placeholder: composite_norm_npy})
        predictions = []
        for j, top_k_row_idx in enumerate(top_k_idx):
            top_k_nn_labels = self.train_labels[top_k_row_idx]
            pred = \
                1 if np.sum(top_k_nn_labels) >= self.prediction_thresh else 0
            predictions.append(pred)
        return predictions

    def train_model(self, X, y, batch_size=None, n_epochs=None):
        self.train_data = X
        self.train_labels = y

class GaussianKernelNearestNeighborModel(ModelWrapper):
    def __init__(self, bandwidth, max_norm_batch_size=10000, name=''):
        self.bandwidth = bandwidth
        self.name = name
        self.train_data = None
        self.train_labels = None
        self.max_norm_batch_size = max_norm_batch_size
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.training_set_tensor = \
            tf.placeholder(tf.float32, shape=[None, None])
        self.test_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
        self.norm_tensor = compute_pairwise_dists(self.training_set_tensor,
                                                  self.test_set_tensor)
        self.norm_tensor_placeholder = \
            tf.placeholder(tf.float32, shape=[None, None])
        self.gaussian_kernel_tensor = \
            tf.exp(-tf.square(self.norm_tensor_placeholder) / bandwidth)
        self.sess.run(self.init_op)

    def predict_floats(self, X):
        if (self.train_data is None) or (self.train_labels is None):
            raise Exception("Train data and labels have not been instantiated")
        composite_norm_npy = run_pairwise_dists(
            self.sess, self.training_set_tensor, self.test_set_tensor,
            self.norm_tensor, self.max_norm_batch_size, self.train_data, X)
        kernel_weights = self.sess.run(
            self.gaussian_kernel_tensor,
            feed_dict={self.norm_tensor_placeholder: composite_norm_npy})
        predictions = []
        for j, ex_weights in enumerate(kernel_weights):
            y_hat = ex_weights * self.train_labels
            pred = 1 if y_hat > 0.5 else 0
            predictions.append(pred)
        return predictions

    def train_model(self, X, y, batch_size=None, n_epochs=None):
        self.train_data = X
        self.train_labels = y
