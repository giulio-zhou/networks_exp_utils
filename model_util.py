import caffe
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

class CaffeModel(ModelWrapper):
    def __init__(self, caffe_prototxt_path=None,
                 caffemodel_path=None, solver_prototxt_path=None):
        self.caffe_prototxt_path = caffe_prototxt_path
        self.caffemodel_path = caffemodel_path
        self.solver_prototxt_path = solver_prototxt_path
        self.model = None

    def load_model(self):
        self.model = caffe.Net(
                self.caffe_prototxt_path, self.caffemodel_path, caffe.TEST)

    def predict_floats(self, X):
        self.model.blobs['data'].reshape(*X.shape)
        self.model.blobs['label'].reshape(X.shape[0])
        self.model.blobs['data'].data[...] = X
        softmax_outputs = self.model.forward()
        print(softmax_outputs)
        softmax_outputs = softmax_outputs['softmax'][:, 1]
        return softmax_outputs

    def predict_ints(self, X):
        return self.predict_floats(X)

    def save_model(self, paths):
        caffemodel_path = paths[0]
        self.model.save(caffemodel_path)

    def train_model(self, X, y):
        solver = caffe.AdamSolver(self.solver_prototxt_path)
        solver.net.blobs['data'].reshape(*X.shape)
        solver.net.blobs['label'].reshape(X.shape[0])
        solver.net.blobs['data'].data[...] = X
        solver.net.blobs['label'].data[...] = y
        solver.solve()
        self.model = solver.net

# Define class of simple Tensorflow models.
# These are intended for simple binary classification tasks.
class TensorflowSimpleModel(ModelWrapper):
    def __init__(self, model_name=None, model_path=None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        # Start session and construct graph.
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, 1024])
        self.label_tensor = tf.placeholder(tf.float32, shape=[None])
        self.ex_weight_tensor = tf.placeholder(tf.float32, shape=[None])
        model_fn = get_simple_tf_model_by_name(self.model_name)
        self.loss, self.classes, self.probabilities, self.accuracy = model_fn(
            self.input_tensor, self.label_tensor, self.ex_weight_tensor)
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()

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
        self.__init__(self.model_name, self.model_path)

    def save_model(self, paths):
        pass

    def train_model(self, X, y, ex_weights=None):
        if ex_weights is None:
            ex_weights = np.ones([len(y)])
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(self.loss)
        batch_size = 64
        num_iter = int(5 * len(y) / batch_size)
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
        batch_num_pos = int(class_balance[1] * batch_size)
        batch_num_neg = batch_size - batch_num_pos
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        for i in range(num_iter):
            # Create balanced input batches.
            if batch_num_pos == 0:
                idx = np.random.choice(neg_idx, batch_num_neg, replace=False)
                X_feed, y_feed = X[idx], y[idx]
                alpha_feed = ex_weights[idx]
            elif batch_num_neg == 0:
                idx = np.random.choice(pos_idx, batch_num_pos, replace=False)
                X_feed, y_feed = X[idx], y[idx]
                alpha_feed = ex_weights[idx]
            else:
                pos_ex_idx = \
                    np.random.choice(pos_idx, batch_num_pos, replace=False)
                neg_ex_idx = \
                    np.random.choice(neg_idx, batch_num_neg, replace=False)
                all_idx = np.hstack([neg_ex_idx, pos_ex_idx])
                X_feed, y_feed = X[all_idx], y[all_idx]
                alpha_feed = ex_weights[all_idx]
            _, loss, accuracy = \
                self.sess.run([train, self.loss, self.accuracy],
                              feed_dict={self.input_tensor: X_feed,
                                         self.label_tensor: y_feed,
                                         self.ex_weight_tensor: alpha_feed})
            if i % 1 == 0:
                print("@{} - loss: {}, accuracy: {}".format(i, loss, accuracy))
        save_path = self.saver.save(self.sess, self.model_path)
        print("Model saved in file: %s" % save_path)

def simple_classifier(inputs, labels, ex_weights):
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    onehot_labels = tf.reshape(onehot_labels, [-1, 2])
    # Network layers.
    hidden = tf.layers.dense(inputs=inputs, units=200, activation=tf.nn.relu)
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
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=classes)
    return loss, classes, probabilities, accuracy

def get_simple_tf_model_by_name(model_name):
    if model_name == 'simple_classifier':
        model_fn = simple_classifier
    else:
        print("No valid model named %s" % model_name)
        exit(1)
    return model_fn
