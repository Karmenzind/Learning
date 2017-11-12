# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv1D, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger
from keras.utils import plot_model


def preprocess(data, col, predict_step=1, win=100, istrain=True):
    if istrain:
        dd = data.ix[:, col].values
        x = []
        y = []
        n = dd.shape[0]
        for i in range(n - win - predict_step + 1):
            x.append(dd[i:i + win].reshape((1, -1)))
            y.append(dd[i + win + predict_step - 1].reshape((1, 1)))

        try:
            x = np.concatenate(x)
            y = np.concatenate(y)
        except:
            x = []
            y = []
        return x, y

    if data.shape[0] < win:
        return [], []

    dd = data.ix[:, col].values
    n = dd.shape[0]
    x = dd[n - win:].reshape((1, -1))
    lastx = dd[n - 1]
    return x, lastx


def make_model(input_dim=100):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(input_dim,)))
    model.add(Reshape((100, 1)))
    model.add(Conv1D(10, 10, padding='valid', activation='relu'))
    model.add(Conv1D(10, 10, padding='valid', activation='relu'))
    model.add(Reshape((-1,)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='Adadelta')
    return model


def train(arg):
    x = []
    y = []
    for f in os.listdir(arg.datapath):
        data = pd.read_excel(os.path.join(arg.datapath, f))
        for i in range(4):
            xx, yy = preprocess(data,
                                i + 1,
                                predict_step=arg.predictStep,
                                win=100)

            if len(xx) > 0:
                x.append(xx)
                y.append(yy)

    x = np.concatenate(x)
    y = np.concatenate(y)

    trainN = int(x.shape[0] * 0.9)
    id = np.arange(x.shape[0])
    np.random.shuffle(id)
    trainX = x[id[:trainN]]
    trainY = y[id[:trainN]]

    testX = x[id[trainN:]]
    testY = y[id[trainN:]]

    model = make_model()

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

    x = []
    y = []
    llbname = []

    for f in os.listdir(arg.datapath):
        data = pd.read_excel(os.path.join(arg.datapath, f))
        if data.shape[0] < 100:
            continue
        llbname.append(f)
        lastinput = []
        predict = []
        for i in range(4):
            xx, xlast = preprocess(data,
                                   i + 1,
                                   predict_step=arg.predictStep,
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
    with open(arg.output, 'w') as fh:
        for i, ll in enumerate(result.tolist()):
            items = ','.join(str(xx) for xx in ll)
            line = '{},{}\n'.format(llbname[i], items)
            fh.write(line)


def draw():
    model = make_model(input_dim=100)
    plot_model(model, to_file='model.png', show_shapes=True)


def parse_cmd():
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
    if arg.cmd == 'train':
        train(arg)
    elif arg.cmd == 'test':
        test(arg)
    elif arg.cmd == 'draw':
        draw()
