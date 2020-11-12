import tkinter as tk
import math
import numpy as np
import os
import matplotlib.pyplot as plt

# Function set
# Calculate e(only for output layer)
def get_error(d, y):
    return d-y
# Calculate sigmoidal(v) = y
def sigmoidal(v):
    if v < -100:return 0
    else:return 1/(1+math.exp(-v))
# Calculate sigmoidal'(v) = dy/dv
def sigmoidal_diff(v):
    return sigmoidal(v)*(1-sigmoidal(v))
# Calculate output layer gradient
def output_gradient(d, v):
    return get_error(d, sigmoidal(v))*sigmoidal_diff(v)
# Calculate hidden layer gradient
def hidden_gradient(v, grad_mul_w_sum):
    return sigmoidal_diff(v)*grad_mul_w_sum
# Calculate delta_w(add momentum version)
def get_delta_w(momentum, w, lr, gradient, v):
    return momentum*w + lr*gradient*sigmoidal(v)

# Neural network implementation
class Net():
    # init net(n*n*1)
    def __init__(self, x_num, momentum, lr):
        print('Net __init__ init')
        self.xinput_v = np.zeros([x_num])
        self.hidden_v = np.zeros([x_num])
        self.output_v = 0.0
        self.hidden_grad = np.zeros([x_num])
        self.output_grad = 0.0
        self.w_to_hidden = np.random.random([x_num, x_num])
        self.w_to_output = np.random.random([x_num])

        self.x_num = x_num
        self.momentum = momentum
        self.lr = lr

    # Check x shape, if ok then call forward and backward and fit_weight
    def doit_just_doit(self, x, y):
        #print('Net doit_just_doit init')
        if self.x_num == x.shape[0]:
            self.forward(x)
            self.backward(y)
            self.fit_weight()
        else:print('fit shape error')

    # Calculate every node v
    def forward(self, x):
        self.xinput_v = x
        # xinput->hidden ok
        for hidden_idx in range(self.x_num):
            v = 0
            for xinput_idx in range(self.x_num):
                v += self.w_to_hidden[hidden_idx, xinput_idx] * sigmoidal(self.xinput_v[xinput_idx])
            self.hidden_v[hidden_idx] = v
        # hidden->output ok
        v = 0
        for hidden_idx in range(self.x_num):
            v += self.w_to_output[hidden_idx] * sigmoidal(self.hidden_v[hidden_idx])
        self.output_v = v

    # Calculate every node gradient
    def backward(self, y):
        # output
        v = self.output_v
        self.output_grad = output_gradient(y, v)
        # hidden
        for hidden_idx in range(self.x_num):
            grad_mul_w_sum = self.output_grad * self.w_to_output[hidden_idx]
            self.hidden_grad[hidden_idx] = hidden_gradient(self.hidden_v[hidden_idx], grad_mul_w_sum)
        #print('grad:', self.output_grad, self.hidden_grad)

    # Fix every weight
    def fit_weight(self):
        # to_hidden
        for hidden_idx in range(self.x_num):
            grad = self.hidden_grad[hidden_idx]
            for xinput_idx in range(self.x_num):
                v = self.xinput_v[xinput_idx]
                w = self.w_to_hidden[hidden_idx, xinput_idx]
                delta_w = get_delta_w(self.momentum, w, self.lr, grad, v)
                self.w_to_hidden[hidden_idx, xinput_idx] = w + delta_w
        # to_output
        grad = self.output_grad
        for hidden_idx in range(self.x_num):
            v = self.hidden_v[hidden_idx]
            w = self.w_to_output[hidden_idx]
            delta_w = get_delta_w(self.momentum, w, self.lr, grad, v)
            self.w_to_output[hidden_idx] = w + delta_w

    # Calculate Eav
    def get_eav(self, x_dataset, y_dataset):
        N = x_dataset.shape[0]
        Eav = 0
        for idx in range(x_dataset.shape[0]):
            self.forward(x_dataset[idx])
            e = get_error(y_dataset[idx], sigmoidal(self.output_v))
            # for one output
            E = e**2 * 0.5
            Eav += E
        Eav = Eav/N
        return Eav

    # Calculate recognition rate
    def get_recognition_rate(self, x_dataset, y_dataset):
        true_one, true_zero, false_one, false_zero = 0, 0, 0, 0
        for x_idx in range(x_dataset.shape[0]):
            self.forward(x_dataset[x_idx])
            if self.output_v > 0.5 and y_dataset[x_idx] == 1.0:true_one+=1
            elif self.output_v > 0.5 and y_dataset[x_idx] == 0.0:false_one+=1
            elif self.output_v < 0.5 and y_dataset[x_idx] == 0.0:true_zero+=1
            elif self.output_v < 0.5 and y_dataset[x_idx] == 1.0:false_zero+=1
        return (true_one+true_zero)/(true_one+true_zero+false_one+false_zero)

    # Nonlinear trans(input x -> hidden y)
    def get_nonlinear_transformer(self, x_dataset):
        #print('before', x_dataset[-1])
        nonlinear_x_dataset = np.empty_like(x_dataset)
        for x_idx in range(nonlinear_x_dataset.shape[0]):
            for y_idx in range(nonlinear_x_dataset.shape[1]):
                nonlinear_x_dataset[x_idx, y_idx] = sigmoidal(np.dot(self.w_to_hidden[y_idx], x_dataset[x_idx]))
        #print('after', nonlinear_x_dataset[-1])
        #print('compare', self.hidden_v)
        return nonlinear_x_dataset


# Input dataset, randomly split to X and label * training(2/3) and testing(1/3)
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

# Plot point and weight line
def plot(dataset, w, X_train, label_train, X_test, label_test, file_name_we_want, if_nonlinear):
    training_num = X_train.shape[0]
    testing_num = X_test.shape[0]

    # for train
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

    # for test
    for j in range(testing_num):
        if label_test[j] == 0.0:
            plt.scatter(X_test[j, 1], X_test[j, 2], c='r', marker='x', s=10)
        elif label_test[j] == 1.0:
            plt.scatter(X_test[j, 1], X_test[j, 2], c='g', marker='x', s=10)
        elif label_test[j] == 2.0:
            plt.scatter(X_test[j, 1], X_test[j, 2], c='b', marker='x', s=10)
        else :
            plt.scatter(X_test[j, 1], X_test[j, 2], c='black', marker='x', s=10)

    # case split
    if if_nonlinear:
        plt.plot([0, 1], [linear_equations(w, 0), linear_equations(w, 1)])
        plt.title(file_name_we_want+'(nonlinear)'+'\npoint label: 0=red, 1=green, 2=blue, others=black; o=train, x=test')
        plt.savefig(os.path.abspath('.') + '\\' + file_name_we_want.split('.')[0]+'(nonlinear)')
    else:
        plt.plot([np.min(dataset[:, 1]), np.max(dataset[:, 1])], [linear_equations(w, np.min(dataset[:, 1])), linear_equations(w, np.max(dataset[:, 1]))])
        plt.title(file_name_we_want+'(linear)'+'\npoint label: 0=red, 1=green, 2=blue, others=black; o=train, x=test')
        plt.savefig(os.path.abspath('.') + '\\' + file_name_we_want.split('.')[0]+'(linear)')
    plt.xlabel('w0:{:.4f}  w1:{:.4f}  w2:{:.4f}    linear_equations: y = -({:.4f}*x - ({:.4f}))/{:.4f}'.format(w[0], w[1], w[2], w[0], w[1], w[2]))
    plt.show()
# Input weight w and point x, return y (w1x+w2y+w0=0 --> y=-(w0w1x)/w2)
def linear_equations(w, x):
    return -(w[1]*x - w[0])/w[2]


# Run when button be pushed.
def main():
    print('main init')

    # Get dataset_list and create dataset_list_2D2G
    data_path = os.path.join(os.path.abspath('.'), 'DataSet')
    for root, dirs, dataset_list in os.walk(data_path):pass
    #dataset_list_2D2G = ['perceptron1.txt', 'perceptron2.txt', '2Ccircle1.txt', '2Circle1.txt', '2Circle2.txt', '2CloseS.txt', '2CloseS2.txt', '2CloseS3.txt', '2cring.txt', '2CS.txt', '2Hcircle1.txt', '2ring.txt']

    # Get and check and set file_name_we_want, then set file_name
    file_name_we_want = file_name_entry.get()
    if not(file_name_we_want in dataset_list):
        file_name_we_want = '2cring.txt'
        print('file_name_we_want not in basic dataset, change to 2cring.txt')
    path_file_name = os.path.join(data_path, file_name_we_want)
    dataset = np.loadtxt(path_file_name, delimiter=' ')
    print(file_name_we_want, ', dataset shape:', dataset.shape)

    # Set learning rate and convergence condition(max try times)
    try:learning_rate = float(learning_rate_entry.get())
    except ValueError:learning_rate = 0.01
    try:convergence_condition = int(convergence_condition_entry.get())
    except ValueError:convergence_condition = 2000
    expect_error = 0.01
    momentum = 0
    print('learning_rate, convergence_condition, expect_error, momentum:', learning_rate, convergence_condition, expect_error, momentum)

    # Set X_train, label_train, X_test, label_test and rebuild X_train and X_test(add threshold = -1)
    threshold = np.full((dataset.shape[0], 1), -1.0) # threshold default = -1
    dataset = np.hstack((threshold, dataset))
    X_train, label_train, X_test, label_test = split_dataset(dataset)
    print('dataset shape:', X_train.shape, label_train.shape, X_test.shape, label_test.shape)

    # Create sorted labels list, and create normalized labels
    labels = []
    for item in dataset[:, -1]:
        if item not in labels:labels.append(item)
    sorted(labels)
    print('labels:', labels)
    normalized_labels = []
    for label in labels:
        normalized_labels.append( (label-min(labels))/(max(labels)-min(labels)) )
    print('normalized_labels:', normalized_labels)
    normalized_label_train = (label_train-min(labels))/(max(labels)-min(labels))
    normalized_label_test = (label_test-min(labels))/(max(labels)-min(labels))


    # Init net
    net = Net(x_num=X_train.shape[1], momentum=momentum, lr=learning_rate)
    counter = 0
    old_train_Eav = 1.0
    old_test_Eav = 1.0
    while True:
        # One dataset run
        for idx in range(X_train.shape[0]):
            net.doit_just_doit(X_train[idx], normalized_label_train[idx])

        # Every 100 epochs
        #['perceptron1.txt', 'perceptron2.txt', '2Ccircle1.txt', '2Circle1.txt', '2Circle2.txt', '2CloseS.txt', '2CloseS2.txt', '2CloseS3.txt', '2cring.txt', '2CS.txt', '2Hcircle1.txt', '2ring.txt']
        if counter%100 == 0:
            print('epochs:', counter)

            print('input:', X_train[-1], normalized_label_train[-1])
            print('v:', net.xinput_v, net.hidden_v, net.output_v)
            print('grad:', net.hidden_grad, net.output_grad)
            print('w:', net.w_to_output, net.w_to_hidden)

            new_train_Eav = net.get_eav(X_train, normalized_label_train)
            new_test_Eav = net.get_eav(X_test, normalized_label_test)
            print('train Eav:', new_train_Eav)
            print('test Eav:', new_test_Eav)
            print()
        counter += 1

        # End condition
        if counter > convergence_condition:
            print('over convergence_condition')
            break
        if (new_train_Eav - old_train_Eav) > 0.0:
            print('train_Eav setback(will become overfitting)')
            break
        if (new_test_Eav - old_test_Eav) > 0.0:
            print('test_Eav setback(will become overfitting)')
            break
        old_train_Eav = new_train_Eav
        old_test_Eav = new_test_Eav
        if new_test_Eav < expect_error:
            print('hey that is pretty good')
            break

    nonlinear_X_train = net.get_nonlinear_transformer(X_train)
    nonlinear_X_test = net.get_nonlinear_transformer(X_test)
    plot(dataset, net.w_to_output, X_train, normalized_label_train, X_test, normalized_label_test, file_name_we_want, if_nonlinear=False)
    plot(dataset, net.w_to_output, nonlinear_X_train, normalized_label_train, nonlinear_X_test, normalized_label_test, file_name_we_want, if_nonlinear=True)

    result = '計算結果：'
    result_label.configure(text=result)

    train_result = '訓練辨識率：{:.4f}'.format(net.get_recognition_rate(X_train, normalized_label_train))
    train_result_label.configure(text=train_result)
    test_result = '測試辨識率：{:.4f}'.format(net.get_recognition_rate(X_test, normalized_label_test))
    test_result_label.configure(text=test_result)

    train_Eav = '訓練均方差：{:.4f}'.format(net.get_eav(X_train, normalized_label_train))
    train_Eav_label.configure(text=train_Eav)
    test_Eav = '測試均方差：{:.4f}'.format(net.get_eav(X_test, normalized_label_test))
    test_Eav_label.configure(text=test_Eav)

    weight_result = '鍵結值[w0, w1, w2]：[{:.4f}, {:.4f}, {:.4f}]'.format(net.w_to_output[0], net.w_to_output[1], net.w_to_output[2])
    weight_result_label.configure(text=weight_result)

    print('main over')


# GUI interface
if __name__ == '__main__':
    print('init')
    window = tk.Tk()
    window.title('input learning rate and convergence condition')
    window.geometry('800x600')
    window.configure(background='white')

    header_label = tk.Label(window, text='多層感知機(採用最小均方法)')
    header_label.pack()

    learning_rate_frame = tk.Frame(window)
    learning_rate_frame.pack(side=tk.TOP)
    learning_rate_label = tk.Label(learning_rate_frame, text='學習率(default = 0.01)')
    learning_rate_label.pack(side=tk.LEFT)
    learning_rate_entry = tk.Entry(learning_rate_frame)
    learning_rate_entry.pack(side=tk.LEFT)

    convergence_condition_frame = tk.Frame(window)
    convergence_condition_frame.pack(side=tk.TOP)
    convergence_condition_label = tk.Label(convergence_condition_frame, text='收斂條件(最大訓練epochs)(default = 2000)')
    convergence_condition_label.pack(side=tk.LEFT)
    convergence_condition_entry = tk.Entry(convergence_condition_frame)
    convergence_condition_entry.pack(side=tk.LEFT)

    file_name_frame = tk.Frame(window)
    file_name_frame.pack(side=tk.TOP)
    file_name_label = tk.Label(file_name_frame, text='要執行的檔案名稱(default = 2cring.txt)')
    file_name_label.pack(side=tk.LEFT)
    file_name_entry = tk.Entry(file_name_frame)
    file_name_entry.pack(side=tk.LEFT)

    calculate_btn = tk.Button(window, text='按下去開始算', command=main)
    calculate_btn.pack()

    result_label = tk.Label(window)
    result_label.pack()
    train_result_label = tk.Label(window)
    train_result_label.pack()
    test_result_label = tk.Label(window)
    test_result_label.pack()
    train_Eav_label = tk.Label(window)
    train_Eav_label.pack()
    test_Eav_label = tk.Label(window)
    test_Eav_label.pack()
    weight_result_label = tk.Label(window)
    weight_result_label.pack()

    window.mainloop()