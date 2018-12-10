import tensorflow as tf
import random
import os
import shutil


class LSTM(object):
    def __init__(self, config, train_data, scope_name):
        self.time_step = config['time_step']
        self.rnn_unit = config['hidden_unit']
        self.batch_size = config['batch_size']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.lr = config['learning_rate']
        self.epochs = config['epochs']
        self.scope_name = scope_name

        self.model_path = 'data/tmp'
        self.model_file = os.path.join(self.model_path, 'model.ckpt')
        self.sess = None
        self.pred = None

        self.train_data = train_data
        self.train_x = []
        self.train_y = []

        self.X = tf.placeholder(tf.float32,
                                [None, self.time_step, self.input_size])
        self.Y = tf.placeholder(tf.float32,
                                [None, self.time_step, self.output_size])

        self.weights = {
            'in': tf.Variable(tf.random_normal([self.input_size, self.rnn_unit])),
            'out': tf.Variable(tf.random_normal([self.rnn_unit, self.output_size]))}
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.rnn_unit, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.output_size, ]))}

        for i in range(len(self.train_data) - self.time_step - 1):
            x = self.train_data[i : i+self.time_step, :]
            y = self.train_data[i+1 : i+self.time_step+1, :]
            self.train_x.append(x.tolist())
            self.train_y.append(y.tolist())

    def _lstm(self, X, scope):
        batch_size, time_step = tf.shape(X)[0], tf.shape(X)[1]
        w_in = self.weights['in']
        b_in = self.biases['in']
        input_org = tf.reshape(self.X, [-1, self.input_size])
        input_rnn = tf.matmul(input_org, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, self.rnn_unit])

        # cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_unit)
        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit, reuse=None)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)

            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn,
                                                         initial_state=init_state,
                                                         dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, self.rnn_unit])
        w_out = self.weights['out']
        b_out = self.biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

    def _train(self):
        # delete previous model
        for dir in list(os.listdir(self.model_path)):
            file_path = os.path.join(self.model_path, dir)
            if os.path.isfile(file_path):
                # remove file
                os.remove(file_path)
            else:
                # remove dir
                shutil.rmtree(file_path)
        print('clear model file, and start training...')

        pred, _ = self._lstm(self.X, self.scope_name)
        self.pred = pred
        # define loss
        loss = tf.reduce_mean(
            tf.square(tf.reshape(pred, [-1]) - tf.reshape(self.Y, [-1])))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # saver=tf.train.Saver(tf.global_variables(), max_to_keep=3)
        with tf.Session().as_default() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for i in range(self.epochs):
                start = random.randint(0, self.batch_size)
                end = start + self.batch_size
                while end < len(self.train_x):
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={self.X: self.train_x[start:end],
                                                   self.Y: self.train_y[start:end]})
                    start += self.batch_size
                    end = start + self.batch_size
                    if step % 20 == 0:
                        print("epoch-",i, " loss:", loss_)
                        # saver.save(sess, self.model_file, global_step=step)
                    step += 1
            print(self.scope_name + " training has finished")
            self.sess = sess

            # prob = sess.run(pred, feed_dict={self.X: [self.train_x[50]]})
            # print(prob)

    def predict(self, test_x=None):
        # pred, _ = self._lstm(self.X)
        # saver = tf.train.Saver(tf.global_variables())
        # with tf.Session() as sess:
        #     model_file = tf.train.latest_checkpoint(self.model_path)
        #     saver.restore(sess, model_file)
        #     prob = sess.run(pred, feed_dict={self.X: [self.train_x[50]]})
        #     print(prob)
        if not self.sess:
            self._train()
        prob = self.sess.run(self.pred, feed_dict={self.X: [test_x]})
        return prob[0]