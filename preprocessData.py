# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
validation_file= "valid.p"
testing_file = "test.p"

# Load original data
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


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

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
n_classes = len(np.unique(np.concatenate([y_train, y_valid, y_test])))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import matplotlib.pyplot as plt
import cv2
# Visualizations will be shown in the notebook.
#%matplotlib inline

# Count number in each class
# Save the index of the first instance to plot later
"""
sign_cnt = np.zeros(n_classes, dtype=np.uint64)
sign_idx = np.zeros(n_classes, dtype=np.uint64)
for x in range(len(y_train)):
    sign_cnt[y_train[x]] += 1
    sign_idx[y_train[x]] = x
for x in range(n_classes):
    print(x, sign_cnt[x])

plt.xlabel('Sign ID')
plt.ylabel('Sign Count')
plt.bar(np.arange(n_classes), sign_cnt)
plt.show()

max_imgs = max(sign_cnt)
#print(max_imgs)
"""

# Plot the first instance of each class
"""
for x in range(n_classes):
    print(x, " ", sign_cnt[x])
    image = X_train[x].squeeze()
    plt.figure(figsize=(1,1))
    plt.imshow(image)

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image)
print(y_train[index])
"""

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


#nadd = 0
#for x in range(n_classes):
#    tmp = int(max_imgs // sign_cnt[x]) - 1
#    if (tmp > 5):
#        tmp = 5
#    nadd += int(tmp * sign_cnt[x]) + 1
#
#X_train_full = np.zeros([n_train+nadd,32,32,3], dtype=np.uint8)
#y_train_full = np.zeros([n_train+nadd], dtype=np.uint8)
#dst = 0
#for x in range(n_train):
#    X_train_full[dst] = X_train[x]
#    y_train_full[dst] = y_train[x]
#    dst += 1
#    tmp = int(max_imgs // sign_cnt[y_train[x]]) - 1
#    if (tmp > 5):
#        tmp = 5
#    for y in range(tmp) :
#        dx = random.randint(-2,2)
#        dy = random.randint(-2,2)
#        dr = random.randint(-1500,1500)/100.0
#        ds = random.randint(90,110)/100.0
#        trnMat = np.float32([[1,0,dx],[0,1,dy]])
#        rotMat = cv2.getRotationMatrix2D((16,16), dr, ds)
#        tst = cv2.warpAffine(X_train[x], trnMat, (32,32))
#        tst = cv2.warpAffine(tst, rotMat, (32,32))
#        X_train_full[dst] = tst
#        y_train_full[dst] = y_train[x]
#        dst += 1
#X_train = X_train_full
#y_train = y_train_full
#n_train = len(X_train)

#for x in range(25):
#  plt.subplot(5,5,x+1)
#  plt.imshow(X_train[x])
#plt.show()

#sign_cnt = np.zeros(n_classes, dtype=np.uint64)
#sign_idx = np.zeros(n_classes, dtype=np.uint64)
#for x in range(len(y_train)):
#    sign_cnt[y_train[x]] += 1
#    sign_idx[y_train[x]] = x
#for x in range(n_classes):
#    print(x, sign_cnt[x])
#plt.bar(np.arange(n_classes), sign_cnt)
#plt.show()

#clahe = cv2.createCLAHE() #Contrast Limited Adaptive Histogram Equalization
#X_train_gray[x] = np.reshape( clahe.apply(tmp), [32,32,1] )

X_train_gray = np.zeros([n_train,32,32,1])
X_train_normalized = np.zeros([n_train,32,32,1])
for x in range(n_train):
    # Convert to grayscale
    tmp = cv2.cvtColor(X_train[x], cv2.COLOR_RGB2GRAY)
    X_train_gray[x] = np.reshape(tmp, [32,32,1])
    # Histogram Equalization
    X_train_normalized[x] = np.reshape( cv2.equalizeHist( tmp ), [32,32,1] )
    # Normalize
    X_train_normalized[x] = ( X_train_normalized[x] - 128 ) /128

# Plot an example of a traffic sign image before and after grayscaling.
"""
index = random.randint(0, len(X_train))
image = X_train[index]
plt.imshow(image)
plt.title(y_train[index])
plt.show()
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(image_gray, cmap = 'gray')
plt.title(y_train[index])
plt.show()
"""

# Plot original gray image and histogram equalized image
"""
img = cv2.imread('gray_sign2.png',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('equ_sign2.png',res)
"""    
    
#Plot an original image and an augmented image
"""
image = X_train[0]
plt.imshow(image.squeeze(), cmap = 'gray')
plt.title('original')
plt.show()
image_nm = X_train_normalized[0]
plt.imshow(image_nm.squeeze(), cmap = 'gray')
plt.title('augmented')
plt.show()
"""

X_valid_gray = np.zeros([n_validation,32,32,1])
X_valid_normalized = np.zeros([n_validation,32,32,1])
for x in range(n_validation):
    # Convert to grayscale
    tmp = cv2.cvtColor(X_valid[x], cv2.COLOR_RGB2GRAY)
    X_valid_gray[x] = np.reshape(tmp, [32,32,1])
    # Histogram Equalization
    X_valid_normalized[x] = np.reshape( cv2.equalizeHist( tmp ), [32,32,1] )
    # Normalize
    X_valid_normalized[x] = ( X_valid_normalized[x] - 128 ) /128


X_test_gray = np.zeros([n_test,32,32,1])
X_test_normalized = np.zeros([n_test,32,32,1])
for x in range(n_test):
    # Convert to grayscale
    tmp = cv2.cvtColor(X_test[x], cv2.COLOR_RGB2GRAY)
    X_test_gray[x] = np.reshape(tmp, [32,32,1])
    # Histogram Equalization
    X_test_normalized[x] = np.reshape( cv2.equalizeHist( tmp ), [32,32,1])  
    # Normalize
    X_test_normalized[x] = ( X_test_normalized[x] - 128 ) /128

print('X_train_gray mean:', np.mean(X_train_gray))
print('X_valid_gray mean:', np.mean(X_valid_gray))
print('X_test_gray mean:', np.mean(X_test_gray))
print('X_train_normalized mean and stddev:', np.mean(X_train_normalized), np.std(X_train_normalized))
print('X_valid_normalized mean and stddev:', np.mean(X_valid_normalized), np.std(X_valid_normalized))
print('X_test_normalized mean and stddev:', np.mean(X_test_normalized), np.std(X_test_normalized))


training_file   = "train_gr_eq_nm.p"
validation_file = "valid_gr_eq_nm.p"
testing_file    = "test_gr_eq_nm.p"
X_train = X_train_normalized
X_valid = X_valid_normalized
X_test = X_test_normalized

# Save grayscaled, equalizeHisted and normalized data
with open(training_file, mode='wb') as f:
    pickle.dump({'features':X_train, 'labels':y_train}, f)
with open(validation_file, mode='wb') as f:
    pickle.dump({'features':X_valid, 'labels':y_valid}, f)
with open(testing_file, mode='wb') as f:
    pickle.dump({'features':X_test, 'labels':y_test}, f)


