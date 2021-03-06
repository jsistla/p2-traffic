"""
utility helper functions and class definitions for ConvNet class
"""

import tensorflow as tf
import numpy as np
import math
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class ConvNet(object):
    """
    Tensorflow class for ConvNet adds function definitions 
    for training , calculating the accuracy and the predictions functions
    Defines the following functions
    train: Trains the current model with the given training data, softmax
           CE error measure and ADAM optimizer is used.
    preditions: returns the predictions of the trained model,
                returns one-hot encoded values
    score: Returns the accuracy of the trained model on the provided test data
    _plt_confusion_matrix: Given one-hot encoded labels and preds, displays a confusion matrix
                      Ref: http://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

    conv2d: Adds a 2D Convolutional layer to the model.
            A wrapper on top of tensorflow implementation conv2d
    pool2d: Adds a 2D pooling layer to the model.
            A wrapper on top of tensorflow implementation for max or avg pooling
    fully_connected: Adds a fully connected layer to the model.
    dropout: Adds dropout to the previous layer 
             A wrapper on top of tensorflow implementation
    """
    
    def __init__(self, learning_rate=0.001, batch_size=256, keep_prob=0.5,
                 image_shape=(32,32), color_channels=1, n_classes=43):
        
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size
        self.color_channels = color_channels
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.train_time = None
        
        self._data = tf.placeholder(tf.float32,
                [None, image_shape[0], image_shape[1], color_channels],
                name='data')
        self._labels = tf.placeholder(tf.float32, 
                [None, n_classes], 
                name='labels')
        
        self._dropout = tf.placeholder(tf.float32, name='dropout')
        self._keep_prob = keep_prob
        self._last_layer = 'INPUT'
        
        self.layer_depths = {'INPUT': color_channels}
        self.weights = {}
        self.biases = {}
        self.LOGITS = None
    
    def train(self, train, val, max_epochs=100, l2_beta=0.001,
              threshold=10, save_loc='Checkpoints/model.ckpt',
              OPTIMIZER=tf.train.AdamOptimizer):

        if self.LOGITS is None:
            raise ValueError('Add some layers!')
        
        # Split inputs into images and labels
        X_train, y_train = train
        X_val, y_val = val
        
        # Ensure that input color channels match
        assert X_train.shape[3] == self.color_channels and \
               X_val.shape[3] == self.color_channels, "Color mismatch"
        
        # Ensure that train and val labels are equivalent and in the expected 
        # format.
        assert y_train.ndim > 1 and y_train.ndim == y_val.ndim, \
               "Labels must be one-hot encoded"
        assert y_train.shape[1] == y_val.shape[1], \
               "Train and Val sets have different number of classes."
        assert y_train.shape[1] == self.n_classes, \
               "Different number of classes than what was specified"
        
        # Add an output layer if one doesn't already exist
        if 'OUT' not in self.weights:
            self.fully_connected('OUT', self.n_classes, ACTIVATION=None)
            self._last_layer = 'OUT'
        
        # Define loss and optimizer for training
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                   self.LOGITS,
                   self._labels))
        # Add l2 regularization to the loss
        for key in list(self.weights.keys()):
            loss += l2_beta * tf.nn.l2_loss(self.weights[key])
        
        optimizer = OPTIMIZER(learning_rate=self.LEARNING_RATE)
        optimizer = optimizer.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.LOGITS, 1),
                                      tf.argmax(self._labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            print('Starting training process:')
            
            sess.run(init)
            saver = tf.train.Saver()

            best = {'epoch': 0, 'val_acc': 0, 'last': 0}
            n_obs = X_train.shape[0]
            start_time = time.clock()
            
            for epoch in range(max_epochs):
                # Train model over all batches
                n_examples = 0
                l_running_avg = 0
                for batch_x, batch_y in self._batches(X_train, y_train):
                    n_examples += min(self.BATCH_SIZE, n_obs)

                    l_running_avg += sess.run(
                        [optimizer, loss],
                        feed_dict={self._data: batch_x,
                                   self._labels: batch_y,
                                   self._dropout: self._keep_prob}
                      )[1]

                    print('\r',
                          'Epoch: %03d | %05.1f%% - Loss: %2.9f'
                          % (epoch+1, min(100*n_examples/n_obs, 100.0), l_running_avg*self.BATCH_SIZE/n_examples),
                          end=''
                      )
                
                # Calculate accuracy over validation set
                c = []
                for batch_x, batch_y in self._batches(X_val, y_val, shuffle=False):
                    c.append(sess.run(accuracy,
                                      feed_dict={self._data: batch_x,
                                                 self._labels: batch_y,
                                                 self._dropout: 1.0}))
                
                c = np.mean(c).astype('float32')
                print(" | Validation Acc: %2.4f%%" % (c*100.0), end='')

                best['last'] += 1

                if best['val_acc'] < c:
                    print(' - Best!', end='')
                    best = {'epoch': epoch, 'val_acc': c, 'last': 0}
                    saver.save(sess, save_loc)

                if best['last'] >= threshold:
                    break
                print()
            
            # Calculate runtime and print out results
            self.train_time = time.clock() - start_time
            m, s = divmod(self.train_time, 60)
            h, m = divmod(m, 60)
            print("\nOptimization Finished!! Training time: %02dh:%02dm:%02ds"
                  % (h, m, s))
            print('Best Validation Loss: %2.9f' % best['val_acc'])
    
    def predict(self, X, save_loc='Checkpoints/model.ckpt'):

        assert self._last_layer == 'OUT', "You must train the model first!"
        
        model = tf.nn.softmax(self.LOGITS)
        pred = None
        
        with tf.Session() as sess:
            # Load saved weights
            tf.train.Saver().restore(sess, save_loc)
            for batch_x, _ in self._batches(X, shuffle=False):
                tmp = sess.run(model, feed_dict={self._data: batch_x,
                                                 self._dropout: 1.0})
                if pred is None: pred = tmp
                else: pred = np.concatenate((pred, tmp))
        return pred
    
    def score(self, test_data, plot=False, normalize=False):

        X, y = test_data
        assert X.shape[0] == y.shape[0], "Different number of obs and labels."
        
        count, correct = 0, 0
        pred = self.predict(X)
        
        if plot: self._plt_confusion_matrix(y, pred, normalize=normalize)
        
        for obs in range(X.shape[0]):
            if pred[obs,...].argmax() == y[obs,...].argmax():
                correct += 1
            count += 1
        return correct/count
    
    def conv2d(self, name, kernel_size, depth, input_padding=0, stride=1, 
               ACTIVATION=tf.nn.relu, padding='SAME'):
 
        assert name not in self.weights, "Layer name must be unique."
        
        if ACTIVATION is None: ACTIVATION=lambda x:x
        
        if self.LOGITS is None: INPUT = self._data
        else: INPUT = self.LOGITS
        
        if input_padding:
            INPUT = tf.pad(INPUT, [[0,0],[input_padding,input_padding],
                                   [input_padding,input_padding],[0,0]])
        
        self.layer_depths[name] = depth
        self.weights[name] = tf.Variable(tf.truncated_normal((
            [kernel_size, 
             kernel_size, 
             self.layer_depths[self._last_layer], 
             depth]),
            stddev=0.1),
            name=name)
        self.biases[name] = tf.Variable(tf.zeros(depth), name=name)
        
        strides = [1, stride, stride, 1]
        self.LOGITS = tf.nn.conv2d(INPUT, self.weights[name], strides, padding)
        self.LOGITS = tf.nn.bias_add(self.LOGITS, self.biases[name])
        self.LOGITS = ACTIVATION(self.LOGITS)
        self._last_layer = name
    
    def fully_connected(self, name, depth, ACTIVATION=tf.nn.relu):

        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer first.')
        
        assert name not in self.weights, "Layer name must be unique."
        
        if name == 'OUT' and ACTIVATION is not None:
            raise ValueError('The output layer cannot have an activation function.')
        
        if ACTIVATION is None: ACTIVATION=lambda x:x
        
        self.layer_depths[name] = depth
        
        self.LOGITS = tf.contrib.layers.flatten(self.LOGITS)
        
        # Flatten the output of the last layer. If the last layer was already 
        # flat, it can't get any flatter :)
        self.weights[name] = tf.Variable(tf.truncated_normal(
            [self.LOGITS.get_shape().as_list()[-1],
             depth],
            stddev=0.1),
            name=name)
        self.biases[name] = tf.Variable(tf.zeros(depth), name=name)
        
        self.LOGITS = tf.matmul(self.LOGITS, self.weights[name]) + self.biases[name]
        self.LOGITS = ACTIVATION(self.LOGITS)
        self._last_layer = name
    
    def dropout(self):
        """
        Adds a dropout layer to the current model.
        
        Uses the model defined drop probability.
        """
        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer to the model first.')
        self.LOGITS = tf.nn.dropout(self.LOGITS, self._dropout)
    
    def pool2d(self, method, kernel_size=2, stride=2, padding='VALID'):

        if self.LOGITS is None:
            raise ValueError('Add a ConvLayer to the model first.')
        assert method in ('MAX','AVG'), "Method must be MAX or AVG."
        
        kernel = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]
        
        if method == 'MAX':
            self.LOGITS = tf.nn.max_pool(self.LOGITS, kernel, strides, padding)
        else:
            self.LOGITS = tf.nn.avg_pool(self.LOGITS, kernel, strides, padding)
    
    def _plt_confusion_matrix(self, labels, pred, normalize=False,
                              title='Confusion matrix', cmap=plt.cm.Blues):

        labels = [label.argmax() for label in labels]
        pred = [label.argmax() for label in pred]
        
        cm = confusion_matrix(labels, pred)
        classes = np.arange(self.n_classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        
        plt.figure(figsize=(9,7))
        plt.imshow(cm, interpolation='nearest', aspect='auto', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    def _batches(self, X, y=None, shuffle=True):

        if X.ndim == 3: 
            return [X[np.newaxis, ...], None]
        
        batch_size = self.BATCH_SIZE
        n_obs = X.shape[0]
        n_batches = math.ceil(n_obs/batch_size)
        
        if shuffle:
            X, y = self._shuffle(X, y)

        for batch in range(0, n_batches*batch_size, batch_size):
            batch_x = X[batch:min(n_obs, batch+batch_size)]
            batch_y = 0 if y is None else y[batch:min(n_obs, batch+batch_size)]
            yield batch_x, batch_y
    
    @staticmethod
    def _shuffle(X, y=None):
        """
        Given data (X) and labels (y), randomly shuffles their order.
        """
        X_shuffled, y_shuffled = [],[]
        n_obs = X.shape[0]

        for i in np.random.permutation(n_obs):
            X_shuffled.append(X[i,...])
            
            if y is None: y_shuffled.append(0)
            else: y_shuffled.append(y[i,...])
        return (np.array(X_shuffled), np.array(y_shuffled))
