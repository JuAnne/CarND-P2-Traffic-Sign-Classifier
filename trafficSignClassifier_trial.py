# Input Hyperparameters from command line
# E.g. python trafficSignClassifier.py 6 16 120 84 >conv1_cfg1_out.txt
# E.g. python trafficSignClassifier.py 16 32 400 200 >conv1_cfg2_out.txt

import sys

conv_depth1 = int(sys.argv[1]) #6   16
conv_depth2 = int(sys.argv[2]) #16  32
fc_depth1 = int(sys.argv[3])  #120  400
fc_depth2 = int(sys.argv[4])  #84   200

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Step 1: Dataset Summary & Exploration
import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
"""
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
"""

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline
import random

fig, axes = plt.subplots(2,5, figsize=(20,8))
axes = axes.ravel()
for i in range(10):
    index = random.randint(0, len(X_train))
    image = X_train[index]#.squeeze()
    axes[i].imshow(image)
    axes[i].set_title(y_train[index])
    

###Pre-process the Data Set (normalization, grayscale, etc.)
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# Convert to grayscale
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)


print('X_train_gray mean:', np.mean(X_train_gray))
print('X_valid_gray mean:', np.mean(X_valid_gray))
print('X_test_gray mean:', np.mean(X_test_gray))

# Normalize the datasets.
X_train_normalized = (X_train_gray - 128)/128
X_valid_normalized = (X_valid_gray - 128)/128
X_test_normalized = (X_test_gray - 128)/128

print('X_train_normalized mean and stddev:', np.mean(X_train_normalized), np.std(X_train_normalized))
print('X_valid_normalized mean and stddev:', np.mean(X_valid_normalized), np.std(X_valid_normalized))
print('X_test_normalized mean and stddev:', np.mean(X_test_normalized), np.std(X_test_normalized))

print('Done data normalization!')

# Visualize original vs normalized images
fig, axes = plt.subplots(1,2, figsize=(15, 6))
axes = axes.ravel()

axes[0].imshow(X_train[0].squeeze(), cmap='gray')
axes[0].set_title('original')

axes[1].imshow(X_train_normalized[0].squeeze(), cmap='gray')
axes[1].set_title('normalized')

#Shuffle the training data.
from sklearn.utils import shuffle

X_train_normalized, y_train = shuffle(X_train_normalized, y_train)
print('Done data shuffling!')

print('X_train_normalized shape:', X_train_normalized[0].shape)

"""
X_train_gray mean: 82.677589037
X_valid_gray mean: 83.5564273756
X_test_gray mean: 82.1484603612
X_train_normalized mean: -0.354081335648 0.515701529314
X_valid_normalized mean: -0.347215411128 0.531148605055
X_test_normalized mean: -0.358215153428 0.521595652937
Done data normalization!
Done data shuffling!
X_train_normalized shape: (32, 32, 1)
"""


###Training Model Architecture
import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128
RATE = 0.001
print('Epochs Size:', EPOCHS)
print('Batch Size:', BATCH_SIZE)
print('Learning Rate:', RATE)

print('Conv1 Depth:', conv_depth1)
print('Conv2 Depth:', conv_depth2)
print('FC1 Depth:', fc_depth1)
print('FC2 Depth:', fc_depth2)

from tensorflow.contrib.layers import flatten

def ConvNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    keep_prob = 0.5
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28xconv_depth1.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, conv_depth1), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(conv_depth1))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    #  Pooling. Input = 28x28xconv_depth1. Output = 14x14xconv_depth1.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10xconv_depth2.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, conv_depth1, conv_depth2), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(conv_depth2))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    #  Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10xconv_depth2. Output = 5x5xconv_depth1.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5xconv_depth2. Output = 5*5*conv_depth2.
    # The flatten function flattens a Tensor into two dimensions: (batches, length). The batch size remains unaltered.
    fc0   = flatten(conv2)

    # Dropout
    fc0 = tf.nn.dropout(fc0, keep_prob)
    print("Apply dropout!")
    
    # Layer 3: Fully Connected. Input = 5*5*conv_depth2. Output = fc_depth1.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5*5*conv_depth2, fc_depth1), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(fc_depth1))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = fc_depth1. Output = fc_depth2.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(fc_depth1, fc_depth2), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(fc_depth2))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = fc_depth2. Output = n_classes = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(fc_depth2, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def ConvNet2(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    keep_prob = 0.5
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    #  Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    #  Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional. Input = 5x5x16. Output = 1x1x400.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(400))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

    #  Activation.
    conv3 = tf.nn.relu(conv3)    

    # Flatten. Input = 5x5x16. Output = 400.
    # The flatten function flattens a Tensor into two dimensions: (batches, length). The batch size remains unaltered.
    fc0   = flatten(conv2)
    print("conv2 flatten shape:", fc0.get_shape())
    
    # Flatten. Input = 1x1x400. Output = 400.
    # The flatten function flattens a Tensor into two dimensions: (batches, length). The batch size remains unaltered.
    fc1   = flatten(conv3)
    print("conv3 flatten shape:", fc1.get_shape())
    
    # Layer 4: Concat fc0 and fc1. Input = 400 + 400. Output = 800
    fc2 = tf.concat_v2([fc0, fc1], 1)
    print("fc2 shape:", fc2.get_shape())    

    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)
    print("Apply dropout!")
       
    # Layer 5: Fully Connected. Input = 800. Output = n_classes = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(800, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

"""
conv2 flatten shape: (?, 400)
conv3 flatten shape: (?, 400)
fc2 shape: (?, 800)
"""

###Features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

###Training Pipeline
logits = ConvNet(x)
#logits = ConvNet2(x)   #refer to Sermanet paper               
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = RATE)
training_operation = optimizer.minimize(loss_operation)

###Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
 

###Train, Validate and Test the Model
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

#Run the training data through the training pipeline to train the model.
#Before each epoch, shuffle the training set.
#After each epoch, measure the loss and accuracy of the validation set.
#Save the model after training.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_normalized, y_train = shuffle(X_train_normalized, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_normalized[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        training_accuracy = evaluate(X_train_normalized, y_train)    
        validation_accuracy = evaluate(X_valid_normalized, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
