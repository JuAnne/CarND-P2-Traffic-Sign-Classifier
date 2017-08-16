# Input Hyperparameters from command line
# E.g. python trafficSignClassifier.py 6 16 120 84 >conv1_cfg1_out.txt
# E.g. python trafficSignClassifier.py 16 32 400 200 >conv1_cfg2_out.txt
"""
import sys

conv_depth1 = int(sys.argv[1]) #6   16
conv_depth2 = int(sys.argv[2]) #16  32
fc_depth1 = int(sys.argv[3])  #120  400
fc_depth2 = int(sys.argv[4])  #84   200
img_files = int(sys.argv[5])  #1 2
"""
conv_depth1 = 16
conv_depth2 = 32
fc_depth1 = 400
fc_depth2 = 200
img_files = 2

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

if (img_files == 1):
  training_file = "train_gr_nm.p"
  validation_file= "valid_gr_nm.p"
  testing_file = "test_gr_nm.p"
elif (img_files == 2):
  training_file = "train_gr_eq_nm.p"
  validation_file= "valid_gr_eq_nm.p"
  testing_file = "test_gr_eq_nm.p"
  

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Image Shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

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
#print("Image data shape =", image_shape)
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
    image = X_train[index].squeeze()
    axes[i].imshow(image)
    axes[i].set_title(y_train[index])
    

###Pre-process the Data Set (normalization, grayscale, etc.)
# using preprocessData.py


#Shuffle the training data.
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
print('Done data shuffling!')
print('X_train shape:', X_train[0].shape)


###Training Model Architecture
import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128
RATE = 0.0009
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

    # Pooling. Input = 10x10xconv_depth2. Output = 5x5xconv_depth2.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5xconv_depth2. Output = 5*5*conv_depth2.
    # The flatten function flattens a Tensor into two dimensions: (batches, length). The batch size remains unaltered.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 5*5*conv_depth2. Output = fc_depth1.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5*5*conv_depth2, fc_depth1), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(fc_depth1))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = fc_depth1. Output = 84.
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

###Features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

###Training Pipeline
logits = ConvNet(x)           
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
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        training_accuracy = evaluate(X_train, y_train)    
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    # Calculate Test Accuracy
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    saver.save(sess, './lenet')
    print("Model saved")
 """
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

# reading in an image
import os
#import matplotlib.image as mpimg
import cv2

fig, axes = plt.subplots(1, 6, figsize=(5,2))
axes = axes.ravel()

test_images = []

for i, image in enumerate(os.listdir('new_test_images/')):
    test_image = cv2.imread('new_test_images/' + image)
    axes[i].axis('off')
    axes[i].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    test_images.append(test_image)

#plt.show()    
    
#test_images = np.asarray(test_images)

test_images_gray = np.zeros([len(test_images),32,32,1])
test_images_normalized = np.zeros([len(test_images),32,32,1])

for i in range(len(test_images)):
    # Convert to grayscale
    tmp = cv2.cvtColor(test_images[i], cv2.COLOR_BGR2GRAY)
    test_images_gray[i] = np.reshape(tmp, [32,32,1])
    # Histogram Equalization
    test_images_normalized[i] = np.reshape(cv2.equalizeHist(tmp), [32,32,1])
    # Normalize
    test_images_normalized[i] = (test_images_normalized[i] -128) /128
    plt.imshow(test_images_normalized[i].squeeze(), cmap = 'gray')
    #plt.show()
print('test_images_normalized shape:', test_images_normalized.shape)


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

#test_labels = [1, 2, 4, 13, 38]
test_labels = [1, 11, 12, 18, 38]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./convnet.meta')
    saver.restore(sess, './convnet')
    test_image_accuracy = evaluate(test_images_normalized, test_labels)
    print("New Test Images Accuracy = {:.3f}".format(test_image_accuracy))


### Output Top 5 Softmax Probabilities For Each Image Found on the Web.
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=5)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./convnet.meta')
    saver.restore(sess, "./convnet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: test_images_normalized})
    my_top_k = sess.run(top_k, feed_dict={x: test_images_normalized})
print (my_top_k)   

fig, axes = plt.subplots(5, 6, figsize=(20,8))

axes = axes.ravel()

for i, test_image in enumerate(test_images):
    axes[6*i].axis('off')
    axes[6*i].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axes[6*i].set_title('{}'.format(test_labels[i]))
    
    guess1 = my_top_k[1][i][0]
    index1 = np.argwhere(y_valid == guess1)[0]   
    axes[6*i+1].axis('off')
    axes[6*i+1].imshow(X_valid[index1].squeeze(), cmap='gray')
    axes[6*i+1].set_title('{} ({:.0f}%)'.format(guess1, 100*my_top_k[0][i][0]))
    
    guess2 = my_top_k[1][i][1]
    index2 = np.argwhere(y_valid == guess2)[0]   
    axes[6*i+2].axis('off')
    axes[6*i+2].imshow(X_valid[index2].squeeze(), cmap='gray')
    axes[6*i+2].set_title('{} ({:.0f}%)'.format(guess2, 100*my_top_k[0][i][1]))       
    guess3 = my_top_k[1][i][2]
    index3 = np.argwhere(y_valid == guess3)[0]   
    axes[6*i+3].axis('off')
    axes[6*i+3].imshow(X_valid[index3].squeeze(), cmap='gray')
    axes[6*i+3].set_title('{} ({:.0f}%)'.format(guess3, 100*my_top_k[0][i][2]))         
    guess4 = my_top_k[1][i][3]
    index4 = np.argwhere(y_valid == guess4)[0]   
    axes[6*i+3].axis('off')
    axes[6*i+3].imshow(X_valid[index4].squeeze(), cmap='gray')
    axes[6*i+3].set_title('{} ({:.0f}%)'.format(guess4, 100*my_top_k[0][i][3]))         
    guess5 = my_top_k[1][i][4]
    index5 = np.argwhere(y_valid == guess5)[0]   
    axes[6*i+4].axis('off')
    axes[6*i+4].imshow(X_valid[index5].squeeze(), cmap='gray')
    axes[6*i+4].set_title('{} ({:.0f}%)'.format(guess5, 100*my_top_k[0][i][4])) 

plt.show()     
