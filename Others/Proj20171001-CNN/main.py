# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv1D, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger
from keras.utils import plot_model


def preprocess(data, col, predictStep=1, win=100, istrain=True):  # 其中，predictStep、win、istrain为关键字参数，各自被赋予了默认值
    """ 预处理函数
        :param data
        :param col
        :param predictStep
        :param win
        :param istrain 是否为训练
    """
    if istrain:
        dd = data.ix[:, col].values  # ix作用为对行和列重新索引
        x = []
        y = []
        n = dd.shape[0]  # 读取矩阵dd第一维度的长度
        for i in range(n - win - predictStep + 1):
            x.append(dd[i:i + win].reshape((1, -1)))
            y.append(dd[i + win + predictStep - 1].reshape((1, 1)))
        try:  # 尝试
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
    """
    model = Sequential()  # 建立一个序贯模型
    model.add(Dense(100, activation='relu', input_shape=(inputDim,)))
    model.add(Reshape((100, 1)))
    model.add(Conv1D(10, 10, padding='valid', activation='relu'))
    # model.add(AveragePooling1D(pool_siz=2, strides=2, padding='valid'))
    model.add(Conv1D(10, 10, padding='valid', activation='relu'))
    model.add(Reshape((-1,)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='Adadelta')
    return model


def train(arg):
    # data
    files = os.listdir(arg.datapath)  # 获取给定的datapath路径下所有文件，为一个包含文件名的列表
    x = []  # 创建x和y两个列表
    y = []
    for f in files:  # 针对files中的每一个文件f
        f_full_path = os.path.join(arg.datapath, f)  # 获得f的绝对路径
        data = pd.read_excel(f_full_path)  # 用pandas提供的读取excel函数read_excel，将f中的内容存储为矩阵data
        for i in range(4):  # 把for包裹的代码块执行四次，四个循环中，i分别为0、1、2、3
            xx, yy = preprocess(data,
                                i + 1,
                                predictStep=arg.predictStep,
                                win=100)  # 此处调用preprocess，传入data，设定col为i+1，predictStep即命令行调用时候传入的值，
            #  TODO:win
            # 此时，xx和yy分别为
            if len(xx) > 0:  # 只有当xx不为空时候
                x.append(xx)  # append方法作用为给列表增加一个元素，也就是将xx放入x
                y.append(yy)

    """
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
    """
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
    csv_logger = CSVLogger('training.log')

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
    if arg.predictStep == 1:
        model.save(arg.modelfile1)
    else:
        model.save(arg.modelfile2)


def test(arg):
    model1 = load_model(arg.modelfile1)
    model2 = load_model(arg.modelfile2)
    # data
    files = os.listdir(arg.datapath)
    x = []
    y = []
    llbname = []
    for f in files:
        data = pd.read_excel(os.path.join(arg.datapath, f))
        if data.shape[0] < 100:
            continue
        llbname.append(f)
        lastinput = []
        predict = []
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
    result = result.tolist()
    result = [[str(xx) for xx in ll] for ll in result]
    fh = open(arg.output, 'w')
    for i in range(len(result)):
        fh.write(llbname[i] + ',' + ','.join(result[i]) + '\n')
    fh.close()


def draw():
    model = make_model(inputDim=100)
    plot_model(model, to_file='model.png', show_shapes=True)


def parse_cmd():
    """ 增加命令行控制
        举例：
        parser.add_argument('--echo', dest='echo', default=10, type=int)
        在命令中输入 --echo 1，则程序中arg的echo属性就会被赋予整形1
        假如不输入，则默认取default，即10
        此函数与主程序功能无关，暂不作详解
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
