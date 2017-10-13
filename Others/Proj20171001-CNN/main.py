# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv1D, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger
from keras.utils import plot_model

"""
阅读的时候，不要从上到下
正确的顺序是：
0   import部分
1   (使用接口部分）
    if __name__ == '__main__'部分
    parse_cmd函数
2   (载入数据、建模和训练过程)
    train函数
    train中间跳至make_model函数
3   (测试，预测，数据输出)
    test函数
    test过程跳至preprocess函数
4   (画图)
    draw函数
"""


def preprocess(data, col, predictStep=1, win=100, istrain=True):
    """ 数据预处理函数
        :param col为数据的列数
        :param predictStep 预测的哪一步
        :param istrain 是否为训练
        :param win 样本预测间隔，即根据当前样本预测间隔100之后的值

        其中，predictStep、win、istrain为关键字参数，各自被赋予了默认值
    """
    if istrain:  # 如果传入的istrain为真
        dd = data.ix[:, col].values  # ix作用为对行和列重新索引
        x = []
        y = []
        n = dd.shape[0]  # 读取矩阵dd第一维度的长度
        for i in range(n - win - predictStep + 1):
            x.append(dd[i:i + win].reshape((1, -1)))  # append的作用是在列表中添加元素
            y.append(dd[i + win + predictStep - 1].reshape((1, 1)))
        try:  # 尝试执行
            x = np.concatenate(x)
            y = np.concatenate(y)
        except:  # 如果捕获到错误
            x = []
            y = []
        return x, y
    else:
        if data.shape[0] < win:
            return [], []
        dd = data.ix[:, col].values
        n = dd.shape[0]
        x = dd[n - win:].reshape((1, -1))
        lastx = dd[n - 1]
        return x, lastx


def make_model(inputDim=100):
    """
    :param inputDim 输入矩阵的维度，具体含义根据层的类型而定

    此处所添加的层均没有传入batch，即采用随机值
    """
    model = Sequential()  # 建立一个keras提供的序贯模型

    # 增加一个稠密层
    # activation为激活函数，此处采用relu(线性修正)
    # 此处的inputdim用作输入层神经元数100
    model.add(Dense(100, activation='relu', input_shape=(inputDim,)))

    # 此层用于矩阵形状调整
    # 第一个参数 (100,1) 即reshape要达到的目标: length为100的序列，通道数为1
    model.add(Reshape((100, 1)))

    # 增加一个一维卷积层
    # 前两个10分别为filters（过滤）和kernel_size（卷积核大小）
    # padding算法可选有三种: same valid causal，关于padding具体含义可以参考 http://www.jianshu.com/p/05c4f1621c7e
    model.add(Conv1D(10, 10, padding='valid', activation='relu'))

    # 此处注释掉的是一个池化层
    # model.add(AveragePooling1D(pool_siz=2, strides=2, padding='valid'))

    # 同上
    model.add(Conv1D(10, 10, padding='valid', activation='relu'))
    model.add(Reshape((-1,)))

    # 最后的输出层
    model.add(Dense(1, activation='linear'))

    # 进行模型编译，配置学习过程参数
    # loss: 误差分析采用均方误差
    # optimizer: 优化器用adadelta（一种自适应学习率调整方法）
    model.compile(loss='mean_squared_error', optimizer='Adadelta')
    return model


def train(arg):
    """
    载入数据进行训练
    """
    # data
    files = os.listdir(arg.datapath)  # 获取给定的datapath路径下所有文件，为一个包含文件名的列表
    x = []  # 创建x和y两个列表
    y = []
    for f in files:  # 针对files中的每一个文件f
        f_full_path = os.path.join(arg.datapath, f)  # 获得f的绝对路径
        data = pd.read_excel(f_full_path)  # 用pandas提供的读取excel函数read_excel，将f中的内容存储为矩阵data
        for i in range(4):  # 把for包裹的代码块执行四次，四个循环中，i分别为0、1、2、3
            # 此处调用preprocess，传入data，设定col为i+1，predictStep即命令行调用时候传入的值，
            xx, yy = preprocess(data,
                                i + 1,
                                predictStep=arg.predictStep,
                                win=100)
            if len(xx) > 0:  # 只有当xx不为空时候
                x.append(xx)  # append方法作用为给列表增加一个元素，也就是将xx放入x
                y.append(yy)

    '''
    np.concatenate函数用于数组拼接，返回拼接后的数组，效果如下：
    axis参数指定拼接的轴，默认为0，即列方向
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.concatenate((a, b.T), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])
    '''
    x = np.concatenate(x)  # 拼接后，赋给x
    y = np.concatenate(y)

    trainN = int(x.shape[0] * 0.9)  # 乘以0.9之后向左取整
    id = np.arange(x.shape[0])  # arrange作用为创建等差数组，长度为shape_0
    np.random.shuffle(id)  # 打乱顺序
    '''
    # a[:b]的意思是，取列表a的前b个元素
    进而，对于矩阵x，x[[0,2]]效果如下：
    >>> x
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> x[[0,2]]
    array([[1, 2],
           [5, 6]])
    '''
    trainX = x[id[:trainN]]
    trainY = y[id[:trainN]]

    testX = x[id[trainN:]]
    testY = y[id[trainN:]]

    # pdb.set_trace()
    # generate model
    model = make_model()

    # fit model
    print("start training model ...")

    # 记录日志
    csv_logger = CSVLogger('training.log')

    '''
    fit: 开始按照给定的epoch训练模型
    前两个参数： 1. 输入的数据，Numpy数组或者由其组成的列表（假如模型有多个输入）
                2. label标签，Numpy数组
    validation_data 为验证数据，此处传入测试集testX和testY
    callback 这一步训练执行完之后所调用的功能
    '''
    model.fit(trainX,
              trainY,
              batch_size=8,
              epochs=5,
              verbose=1,
              validation_data=(testX, testY),
              callbacks=[csv_logger])
    model.fit(trainX,
              trainY,
              batch_size=128,
              epochs=max(1, arg.echo - 5),
              verbose=1,
              validation_data=(testX, testY),
              callbacks=[csv_logger])

    print("save trained model")
    if arg.piredictStep == 1:
        model.save(arg.modelfile1)
    else:
        model.save(arg.modelfile2)


def test(arg):
    """ 测试过程，也就是预测过程
    具体逻辑与训练过程类似
    :param arg是argparser对象，记录输入的命令行参数
    """
    # 载入之间训练好的模型
    model1 = load_model(arg.modelfile1)  # load_model返回一个keras模型对象
    model2 = load_model(arg.modelfile2)
    # data
    files = os.listdir(arg.datapath)  # 获得datapath下所有文件
    x = []
    y = []
    llbname = []
    for f in files:
        data = pd.read_excel(os.path.join(arg.datapath, f))
        if data.shape[0] < 100:
            continue  # continue的含义为：本次迭代到此为止，跳到下一次迭代，即下一个f
        llbname.append(f)
        lastinput = []  # 寄存输入数据
        predict = []  # 存放预测的中间数据
        for i in range(4):
            xx, xlast = preprocess(data,
                                   i + 1,
                                   predictStep=arg.predictStep,
                                   win=100,
                                   istrain=False)
            py1 = model1.predict(xx)
            py2 = model2.predict(xx)
            lastinput.append(xlast)
            predict.append(py1[0, 0])
            predict.append(py2[0, 0])
        x.append(lastinput)
        y.append(predict)

    x = np.array(x)
    y = np.array(y)
    result = np.concatenate((x, y), axis=1)
    result = result.tolist()  # tolist作用为转化成list类型
    result = [[str(xx) for xx in ll] for ll in result]
    fh = open(arg.output, 'w')
    for i in range(len(result)):
        fh.write(llbname[i] + ',' + ','.join(result[i]) + '\n')
    fh.close()


def draw():
    """
    用plot画出模型图,保存为model.png
    """
    model = make_model(inputDim=100)
    plot_model(model, to_file='model.png', show_shapes=True)


def parse_cmd():
    """ 增加命令行控制
        举例：
        parser.add_argument('--echo', dest='echo', default=10, type=int)
        在命令中输入 --echo 1，则程序中arg的echo属性就会被赋予整形1
        假如不输入，则默认取default，即10
        此函数具体结构与主程序功能无关，暂不作详解
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', dest='cmd', help='train or test')
    parser.add_argument('--predictStep', dest='predictStep', type=int)
    parser.add_argument('--modelfile1', dest='modelfile1')
    parser.add_argument('--modelfile2', dest='modelfile2')
    parser.add_argument('--datapath', dest='datapath')
    parser.add_argument('--output', dest='output')
    parser.add_argument('--echo', dest='echo', default=10, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_cmd()
    # pdb.set_trace()
    if arg.cmd == 'train':  # --cmd后接输入为train时
        train(arg)
    elif arg.cmd == 'test':
        test(arg)
    elif arg.cmd == 'draw':
        draw()
