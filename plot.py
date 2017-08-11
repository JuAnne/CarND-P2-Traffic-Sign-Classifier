import matplotlib.pyplot as plt
import numpy as np
import sys

x1 = np.loadtxt('cfg1.out')
x2 = np.loadtxt('cfg2.out')
x3 = np.loadtxt('cfg3.out')
x4 = np.loadtxt('cfg4.out')
x5 = np.loadtxt('cfg5.out')
x6 = np.loadtxt('cfg6.out')
x7 = np.loadtxt('cfg7.out')

plt.plot(x1[:,0], x1[:,2], label = 'config1', color = 'b')
plt.plot(x2[:,0], x2[:,2], label = 'config2', color = 'r')
plt.plot(x3[:,0], x3[:,2], label = 'config3', color = 'g')
plt.plot(x4[:,0], x4[:,2], label = 'config4', color = 'k')
plt.plot(x5[:,0], x5[:,2], label = 'config5', color = 'c')
plt.plot(x6[:,0], x6[:,2], label = 'config6', color = 'm')
plt.plot(x7[:,0], x7[:,2], label = 'config7', color = 'y')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()