from  read import load_mnist
from addnural import NeuralNetMLP
import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('model.pickle', mode='rb') as fp:
    nn = pickle.load(fp)
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.xlabel('Cost')
plt.ylabel('Epochs')
plt.show()

X_train, y_train = load_mnist('/home/yugo/python_learning/learning/12unit/', kind='train')
X_test, y_test = load_mnist('/home/yugo/python_learning/learning/12unit/', kind='t10k')

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0)/ X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))


miscl_img = X_test[y_test != y_test_pred]
correct_lab = y_test[y_test != y_test_pred]
miscl_lab = y_test_pred[y_test !=y_test_pred]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
print("total: %s" % len(X_test))
print("mistake: %s" % len(miscl_img))
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


miscl_img = X_test[y_test == y_test_pred][:25]
correct_lab = y_test[y_test == y_test_pred][:25]
miscl_lab = y_test_pred[y_test ==y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
# nn_check = NeuralNetMLP(n_output=10, n_features=X_train.shape[1],
#                   n_hidden=10,l1=0.0, l2=0.0, epochs=10,
#                   eta=0.001,alpha=0.0, decrease_const=0.0,
#                   minibatches=1, random_state=1)
#
#
# nn_check.fit(X_train[:5], y_train[:5], print_progress=False)
