### Load the images and plot them here.
### Feel free to use as many code cells as needed.

# reading in an image
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

fig, axes = plt.subplots(1, 5, figsize=(5,2))
axes = axes.ravel()

test_images = []

for i, image in enumerate(os.listdir('new_test_images/')):
    test_image = cv2.imread('new_test_images/' + image)
    axes[i].axis('off')
    axes[i].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    test_images.append(test_image)
    
#plt.show()    
test_images = np.asarray(test_images)

test_images_gray = np.zeros([len(test_images),32,32,1])
test_images_normalized = np.zeros([len(test_images),32,32,1])

for i in range(5):
    # Convert to grayscale
    tmp = cv2.cvtColor(test_images[i], cv2.COLOR_RGB2GRAY)
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
    
