from  read import load_mnist
from nural import NeuralNetMLP
import pickle
import numpy as np
import matplotlib.pyplot as plt


X_train, y_train = load_mnist('/home/yugo/python_learning/learning/12unit/', kind='train')
X_test, y_test = load_mnist('/home/yugo/python_learning/learning/12unit/', kind='t10k')

# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1],
                  n_hidden=50,l1=0.0, l2=0.3, epochs=1000,
                  eta=0.001,alpha=0.001, decrease_const=0.00001,
                  shuffle=True, minibatches=50, random_state=1)


nn.fit(X_train, y_train, print_progress=True)

with open('model.pickle', mode='wb') as fp:
    pickle.dump(nn, fp, protocol=2)

with open('model.pickle', mode='rb') as fp:
    nn = pickle.load(fp)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0,2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.show()


y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0)/ X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))
