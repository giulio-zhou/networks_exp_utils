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
    def __init__(self, model_fn, model_path=None, input_dim=1024):
        self.model_fn = model_fn
        self.model_path = model_path
        self.model = None
        self.input_dim = input_dim
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
