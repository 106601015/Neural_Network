import numpy as np

#初始化輸入層與競爭層神經元的連線權值矩陣
def initCompetition(n , m , d):
    #隨機產生0-1之間的數作為權值
    array = np.random.random(size=n * m *d)
    com_weight = array.reshape(n,m,d)
    return com_weight

#計算向量的二範數
def cal2NF(X):
    res = 0
    for x in X:
        res += x*x
    return res ** 0.5

#對資料集進行歸一化處理
def normalize(dataSet):
    old_dataSet = copy(dataSet)
    for data in dataSet:
        two_NF = cal2NF(data)
        for i in range(len(data)):
            data[i] = data[i] / two_NF
    return dataSet , old_dataSet
#對權值矩陣進行歸一化處理
def normalize_weight(com_weight):
    for x in com_weight:
        for data in x:
            two_NF = cal2NF(data)
            for i in range(len(data)):
                data[i] = data[i] / two_NF
    return com_weight

#得到獲勝神經元的索引值
def getWinner(data , com_weight):
    max_sim = 0
    n,m,d = shape(com_weight)
    mark_n = 0
    mark_m = 0
    for i in range(n):
        for j in range(m):
            if sum(data * com_weight[i,j]) > max_sim:
                max_sim = sum(data * com_weight[i,j])
                mark_n = i
                mark_m = j
    return mark_n , mark_m

#得到神經元的N鄰域
def getNeibor(n , m , N_neibor , com_weight):
    res = []
    nn,mm , _ = shape(com_weight)
    for i in range(nn):
        for j in range(mm):
            N = int(((i-n)**2+(j-m)**2)**0.5)
            if N<=N_neibor:
                res.append((i,j,N))
    return res

#學習率函式
def eta(t,N):
    return (0.3/(t+1))* (math.e ** -N)

#SOM演算法的實現
def do_som(dataSet , com_weight, T , N_neibor):
    '''
    T:最大迭代次數
    N_neibor:初始近鄰數
    '''
    for t in range(T-1):
        com_weight = normalize_weight(com_weight)
        for data in dataSet:
            n , m = getWinner(data , com_weight)
            neibor = getNeibor(n , m , N_neibor , com_weight)
            for x in neibor:
                j_n=x[0];j_m=x[1];N=x[2]
                #權值調整
                com_weight[j_n][j_m] = com_weight[j_n][j_m] + eta(t,N)*(data - com_weight[j_n][j_m])
            N_neibor = N_neibor+1-(t+1)/200
    res = {}
    N , M , _ =shape(com_weight)
    for i in range(len(dataSet)):
        n, m = getWinner(dataSet[i], com_weight)
        key = n*M + m
        if res.has_key(key):
            res[key].append(i)
        else:
            res[key] = []
            res[key].append(i)
    return res

#SOM演算法主方法
def SOM(dataSet,com_n,com_m,T,N_neibor):
    dataSet, old_dataSet = normalize(dataSet)
    com_weight = initCompetition(com_n,com_m,shape(dataSet)[1])
    C_res = do_som(dataSet, com_weight, T, N_neibor)
    draw(C_res, dataSet)
    draw(C_res, old_dataSet)

def draw(C , dataSet):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm' , 'd']
    count = 0
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker='o', color=color[count % len(color)], label=i)
        count += 1
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    #資料處理的方法可以參見上一篇部落格——DBSCAN演算法
    dataSet = loadDataSet("dataSet.txt")
    SOM(dataSet,2,2,4,2)