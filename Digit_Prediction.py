import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random


# ------------- Data Loading ---------------
complete_data = pd.read_csv("complete_MNIST_data.csv")
weights_HW4 = np.array(pd.read_csv("Weights.csv",header=None))
data_X = np.array(complete_data.iloc[:, :-1])
data_y = np.array(complete_data.iloc[:, -1])

train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, train_size=0.8)
train_y = train_y.reshape(3999,1)
num_neurons = 143

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(h):
    h = sigmoid_activation(h)
    ds = h * (1 - h)
    return ds


# ------------ Data Preparation -------------
weight_vector_ip_hid = np.random.randn(784, num_neurons)
weight_vector_hid_op = np.random.randn(num_neurons, 784)
deltaw1 = np.zeros((784,num_neurons))
deltaw2 = np.zeros((num_neurons,784))

# ------------- Model Training ---------------
error_train = []
eta = 0.001
epochs = 1000
alpha = 0.65
momentum = 1
e = []
for ep in range(0, epochs):
    k = 0
    t = 0
    # -------------- Forward Pass ---------------------------
    # St Calculation from i/p weights
    st_train_inp_hid = np.dot(train_X, weight_vector_ip_hid)
    # Applying sigmoid activation for hidden layer neurons
    hid_op = sigmoid_activation(st_train_inp_hid)
    # St calculation from hidden weights
    st_train_hid_op = np.dot(hid_op, weight_vector_hid_op)
    # Applying sigmoid activation function for output layer neurons
    yhat = sigmoid_activation(st_train_hid_op)

    # ------------- Back Propogation Algorithm -----------------
    temp_dw2 = deltaw2
    deltaw2 = np.dot(hid_op.T,
                     (train_X - yhat) * der_sigmoid(
                         np.dot(hid_op, weight_vector_hid_op)))

    temp_dw1 = deltaw1
    deltaw1 = np.dot(train_X.T, np.dot((train_X - yhat) * der_sigmoid(np.dot(hid_op, weight_vector_hid_op)),
                                       weight_vector_hid_op.T) * (der_sigmoid(np.dot(train_X, weight_vector_ip_hid))))


    weight_vector_hid_op = weight_vector_hid_op + (eta * deltaw2)
    weight_vector_ip_hid = weight_vector_ip_hid + (eta * deltaw1)

    if ep % 5 == 0:
        for l, m in zip(train_X, yhat):
            k += np.mean((l - m) ** 2)
        k = k / len(train_X)
        error_train.append(k)
        e.append(ep)
        print("Epoch: ",ep,"Train Loss: ",k)

# ------------ Testing the Model------------------
# St Calculation from i/p weights
st_train_inp_hid_test = np.dot(test_X, weight_vector_ip_hid)
# Applying sigmoid activation for hidden layer neurons
hid_op_test = sigmoid_activation(st_train_inp_hid_test)
# St calculation from hidden weights
st_train_hid_op_test = np.dot(hid_op_test, weight_vector_hid_op)
# Applying sigmoid activation function for output layer neurons
yhat_test = sigmoid_activation(st_train_hid_op_test)
for l, m in zip(test_X, yhat_test):
    t += np.sum((l - m) ** 2)
    t = t / len(test_X)
print("\nOver All Test Loss: ", t)

#--------------Plotting the bar plots for Overall training and testing errors----------
fig,axes = plt.subplots()
bar_plot = [(error_train[len(error_train)-1])/len(train_X), t]
bar_width = 0.5
axes.bar(1.5, bar_plot[0], color="b", width=0.5, label="Train Error")
axes.bar(1.5+bar_width, bar_plot[1], color="g", width=0.5, label="Test Error")
plt.xlim(0,3)
axes.set_xlabel("Errors")
plt.suptitle("Bar Chart For Errors")
plt.legend(loc="best")
plt.show()

#--------------Plotting the bar plots for each digit----------
training_errors = []
testing_errors = []

for i in range(0,10):
    locations = np.where(train_y == i)
    digit_data_train = train_X[locations[0],:]
    st_train_inp_hid_digit = np.dot(digit_data_train, weight_vector_ip_hid)
    # Applying sigmoid activation for hidden layer neurons
    hid_op_digit = sigmoid_activation(st_train_inp_hid_digit)
    # St calculation from hidden weights
    st_train_hid_op_digit = np.dot(hid_op_digit, weight_vector_hid_op)
    # Applying sigmoid activation function for output layer neurons
    yhat_digit = sigmoid_activation(st_train_hid_op_digit)
    digit_loss = np.sum(np.square(yhat_digit - digit_data_train))/len(digit_data_train)
    training_errors.append(digit_loss)

    digit_data_test = test_X[np.where(test_y == i)]
    st_test_inp_hid_digit = np.dot(digit_data_test, weight_vector_ip_hid)
    # Applying sigmoid activation for hidden layer neurons
    hid_op_digit_test = sigmoid_activation(st_test_inp_hid_digit)
    # St calculation from hidden weights
    st_test_hid_op_digit = np.dot(hid_op_digit_test, weight_vector_hid_op)
    # Applying sigmoid activation function for output layer neurons
    yhat_digit_test = sigmoid_activation(st_test_hid_op_digit)
    digit_loss_test = np.sum(np.square(yhat_digit_test-digit_data_test))/len(digit_data_test)
    testing_errors.append(digit_loss_test)

fig,axes = plt.subplots()
bar_plot = [training_errors, testing_errors]
indexes = np.arange(len(bar_plot[0]))
br1 = np.arange(len(bar_plot[0]))
br2 = [x + 0.25 for x in br1]
axes.bar(br1, bar_plot[0], color="b", width=0.25, label="Training Error")
axes.bar(br2, bar_plot[1], color="g", width=0.25, label="Testing Error")
axes.set_xticks(indexes+0.25)
axes.set_xticklabels(("0","1","2","3","4","5","6","7","8","9"))
axes.set_xlabel("Errors")
plt.ylabel("Error")
plt.xlabel("Digits")
plt.suptitle("Bar Chart For Performance Measures")
plt.legend(loc="best")
plt.show()

#---------------Plotting Time Series of training loss----------
plt.plot(e,error_train,color="red", label="Train Error")
plt.legend()
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Training Error")
plt.title("Epochs Vs Training Error")
plt.show()

#--------------- Plotting The Features -----------------------
feature_samples = np.random.randint(low=0,high=142,size=(4,5),dtype=int)
fig, ax = plt.subplots(4, 5)
for i in range(0,4):
    for j in range(0,5):
        actual = weight_vector_hid_op[feature_samples[i][j]].reshape(28,28).T
        plt_original = sns.heatmap(actual, ax=ax[i][j], square=True,cbar=False,xticklabels=False,cmap="gray")
        ax[i][j].set_title("Hidden Neuron: %s"%feature_samples[i][j] )
plt.suptitle("Features Weights For Random 20 Neurons")
plt.show()


#------------ Weight Plots from Homework 4-------------------
fig1, ax1 = plt.subplots(4, 5)
for i in range(0,4):
    for j in range(0,5):
        actual1 = weights_HW4[feature_samples[i][j]].reshape(28,28)
        plt_original1 = sns.heatmap(actual1, ax=ax1[i][j], square=True,cbar=False,xticklabels=False,cmap="gray")
        ax1[i][j].set_title("Hidden Neuron: %s" % feature_samples[i][j])
plt.suptitle("Features Weights For Random 20 Neurons From HW 4")
plt.show()


#--------------- Plotting Sample Output ---------------------
test_samples = random.sample(range(1000),8)
j = 0
fig, ax = plt.subplots(2, 8)
for k in range(0,8):
    actual = test_X[test_samples[k]].reshape(28,28).T
    predicted = yhat_test[test_samples[k]].reshape(28,28).T
    plt_original = sns.heatmap(actual,ax=ax[j][k], square=True,cbar=False,yticklabels=False,cmap="gray")
    plt_generated = sns.heatmap(predicted, ax=ax[j+1][k], square=True,cbar=False,yticklabels=False,cmap="gray")
    plt_original.set_xlabel("Actual Image")
    plt_generated.set_xlabel("Generated Image")

plt.suptitle("Actual and Predicted Images")
plt.show()

