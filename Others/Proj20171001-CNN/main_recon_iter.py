# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd

from keras.layers import Dense, Conv1D, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger
from keras.utils import plot_model


def get_file_prefix(f_name):
    return os.path.splitext(f_name)[0]


def get_model_name_by_file(f_name, predict_step):
    model_name = "%s_model_%s.h5" % (get_file_prefix(f_name), predict_step)
    return model_name
    

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


def train(arg):
    for f in os.listdir(arg.datapath):
        x = []
        y = []
        data = pd.read_excel(os.path.join(arg.datapath, f))
        for i in range(4):
            xx, yy = preprocess(data,
                                i + 1,
                                predict_step=arg.predictStep,
                                win=100)

            if len(xx) > 0:
                x.append(xx)
                y.append(yy)
        if not len:
                continue

        x = np.concatenate(x)
        y = np.concatenate(y)

        trainN = int(x.shape[0] * 0.9)
        _id = np.arange(x.shape[0])
        np.random.shuffle(_id)
        trainX = x[_id[:trainN]]
        trainY = y[_id[:trainN]]

        testX = x[_id[trainN:]]
        testY = y[_id[trainN:]]

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

        model_name = get_model_name_by_file(f, arg.predictStep)
        print("Saving trained model as <%s> ..." % model_name)
        model.save(model_name)

    print("Training finished. :) ")


def test(arg):
    for f in os.listdir(arg.datapath):
        model_name_1 = get_model_name_by_file(f, 1)
        model_name_2 = get_model_name_by_file(f, 2)
        model1 = load_model(model_name_1)
        model2 = load_model(model_name_2)

        x = []
        y = []
        llbname = []
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
        f_prefix = get_file_prefix(f)
        output_f_name = '%s_result.csv' % f_prefix
        print("Saving result as <%s> ..." % output_f_name)
        with open(output_f_name, 'w') as fh:
            for i, ll in enumerate(result.tolist()):
                items = ','.join(str(xx) for xx in ll)
                line = '{},{}\n'.format(llbname[i], items)
                fh.write(line)

    print("Testing finished. :) ")


def draw():
    model = make_model(input_dim=100)
    plot_model(model, to_file='model.png', show_shapes=True)


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', dest='cmd', help='train or test')
    parser.add_argument('--predictStep', dest='predictStep', type=int)
    parser.add_argument('--datapath', dest='datapath')
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
