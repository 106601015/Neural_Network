import tkinter as tk
import math
import numpy as np
import os
import matplotlib.pyplot as plt


# input dataset, randomly split to X and label * training and testing
def split_dataset(dataset):
    np.random.shuffle(dataset)

    training_num = int((2/3)*dataset.shape[0])
    X_train = dataset[:training_num, :]
    X_test = dataset[training_num:, :]

    label_train = X_train[:, -1]
    X_train = X_train[:, :-1]
    label_test = X_test[:, -1]
    X_test = X_test[:, :-1]

    return X_train, label_train, X_test, label_test

# plot point(X_train[n] = [-1, ...], but X_test = [...])
def plot(dataset, w, X_train, label_train, X_test, label_test, file_name_we_want):
    training_num = X_train.shape[0]
    testing_num = X_test.shape[0]

    plt.figure()
    for j in range(training_num):
        if label_train[j] == 0.0:
            plt.scatter(X_train[j, 1], X_train[j, 2], c='r', marker='o', s=10)
        elif label_train[j] == 1.0:
            plt.scatter(X_train[j, 1], X_train[j, 2], c='g', marker='o', s=10)
        elif label_train[j] == 2.0:
            plt.scatter(X_train[j, 1], X_train[j, 2], c='b', marker='o', s=10)
        else :
            plt.scatter(X_train[j, 1], X_train[j, 2], c='black', marker='o', s=10)

    for j in range(testing_num):
        if label_test[j] == 0.0:
            plt.scatter(X_test[j, 1], X_test[j, 2], c='r', marker='x', s=10)
        elif label_test[j] == 1.0:
            plt.scatter(X_test[j, 1], X_test[j, 2], c='g', marker='x', s=10)
        elif label_test[j] == 2.0:
            plt.scatter(X_test[j, 1], X_test[j, 2], c='b', marker='x', s=10)
        else :
            plt.scatter(X_test[j, 1], X_test[j, 2], c='black', marker='x', s=10)

    #plt.plot([np.min(dataset[:, 0]), np.max(dataset[:, 0])], [linear_equations(w, np.min(dataset[:, 0])), linear_equations(w, np.max(dataset[:, 0]))])
    plt.xlabel('w0:{:.4f}  w1:{:.4f}  w2:{:.4f}    linear_equations: y = -({:.4f}*x - ({:.4f}))/{:.4f}'.format(w[0], w[1], w[2], w[0], w[1], w[2]))
    plt.title(file_name_we_want+'\npoint label: 0=red, 1=green, 2=blue, others=black; o=train, x=test')
    plt.savefig(os.path.abspath('.') + '\\' + file_name_we_want.split('.')[0])
    plt.show()

# input weight w and point x, return y (w1x+w2y+w0=0 --> y=-(w0w1x)/w2)
#def linear_equations(w, x):
#    return -(w[1]*x - w[0])/w[2]





# runs when button be pushed.
def main():
    print('main init')

    # dataset_list is list of file names of dataset
    data_path = os.path.join(os.path.abspath('.'), 'DataSet')
    for root, dirs, dataset_list in os.walk(data_path):pass
    #print(dataset_list)
    dataset_list_2D2G = ['perceptron1.txt', 'perceptron2.txt', '2Ccircle1.txt', '2Circle1.txt', '2Circle2.txt', '2CloseS.txt', '2CloseS2.txt', '2CloseS3.txt', '2cring.txt', '2CS.txt', '2Hcircle1.txt', '2ring.txt']
    # get and check and set file_name_we_want, then set file_name
    file_name_we_want = file_name_entry.get()
    if not(file_name_we_want in dataset_list):
        file_name_we_want = '2cring.txt'
        print('file_name_we_want not in basic dataset, change to 2cring.txt')
    path_file_name = os.path.join(data_path, file_name_we_want)
    dataset = np.loadtxt(path_file_name, delimiter=' ')
    print(file_name_we_want, ', dataset shape:', dataset.shape)

    # create sorted labels list, and load dimensions
    labels = []
    for item in dataset[:, -1]:
        if item not in labels:
            labels.append(item)
    sorted(labels)
    dimensions = dataset.shape[1]-1
    print('labels:', labels, 'dimensions:', dimensions)
    # set learning rate and convergence condition(max try times)
    try:learning_rate = float(learning_rate_entry.get())
    except ValueError:learning_rate = 0.01
    try:convergence_condition = int(convergence_condition_entry.get())
    except ValueError:convergence_condition = 1000
    expect_error = 0.001
    print('learning_rate, convergence_condition, expect_error:', learning_rate, convergence_condition, expect_error)

    # randomly init weight(dimensions+1(threshold))
    w = np.random.random((dimensions+1))

    # set X_train, label_train, X_test, label_test and rebuild X_train and X_test(add threshold = -1)
    X_train, label_train, X_test, label_test = split_dataset(dataset)
    #print('dataset shape:', X_train.shape, label_train.shape, X_test.shape, label_test.shape)
    threshold = np.full((X_train.shape[0], 1), -1.0) # threshold default = -1
    X_train = np.hstack((threshold, X_train))
    threshold = np.full((X_test.shape[0], 1), -1.0) # threshold default = -1
    X_test = np.hstack((threshold, X_test))
    #print(w, X_train.shape, X_test.shape, label_train.shape, label_test.shape)
    '''
    # main mse function
    n = 0
    while True:
        err = 0
        for index in range(X_train.shape[0]):
            #print('w, label_train[index], X_train[index] ----------->', w, label_train[index], X_train[index])
            w, e = renew_w(w, label_train[index], X_train[index], learning_rate, labels)
            err += pow(e, 2)
        err/=float(X_train.shape[0])
        n += 1
        #print('w, err', w, err)
        if err < expect_error or n > convergence_condition: break


    result = '計算結果：'
    result_label.configure(text=result)

    fail_pre_num = 0
    for index in range(X_train.shape[0]):
        if get_expected_d(w, X_train[index], labels) != label_train[index]:fail_pre_num += 1
    train_result = '訓練辨識率：{:.4f}%'.format((X_train.shape[0]-fail_pre_num)*100/X_train.shape[0])
    train_result_label.configure(text=train_result)

    fail_pre_num = 0
    for index in range(X_test.shape[0]):
        if get_expected_d(w, X_test[index], labels) != label_test[index]:fail_pre_num += 1
    test_result = '測試辨識率：{:.4f}%'.format((X_test.shape[0]-fail_pre_num)*100/X_test.shape[0])
    test_result_label.configure(text=test_result)

    weight_result = '鍵結值[w0, w1, w2]：[{:.4f}, {:.4f}, {:.4f}]'.format(w[0], w[1], w[2])
    weight_result_label.configure(text=weight_result)
    '''
    plot(dataset, w, X_train, label_train, X_test, label_test, file_name_we_want)

    print('main over')