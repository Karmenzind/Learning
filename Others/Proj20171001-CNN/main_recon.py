# coding: utf-8
# last update: 2017-11-22T13:27:10,129663114+08:00
# modified by: K

import os
import argparse
import time
import numpy as np
import pandas as pd

from keras.layers import Dense, Conv1D, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger
from keras.utils import plot_model

TIME_TAG = time.strftime('%Y-%m-%dT%H:%M:%S')
OUTPUT_FILE = 'result_%s.csv' % TIME_TAG
TRAINING_LOG_FILE = 'training_%s.log' % TIME_TAG
PNG_FILE = 'model_%s.png' % TIME_TAG


# --------------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------------

def get_file_prefix(f_name):
    return os.path.splitext(f_name)[0]


def get_model_path_by_file(arg, f_path, predict_step):
    f_name = os.path.basename(f_path)
    model_name = "%s_model_%s.h5" % (get_file_prefix(f_name), predict_step)
    model_path = os.path.join(arg.modelpath, model_name)
    return model_path


def gen_file_path(dir_path):
    for f in os.listdir(dir_path):
        f_path = os.path.join(dir_path, f)
        if os.path.isfile(f_path):
            yield f_path


# --------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------

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


def train(arg, data, f_path, predict_step):
    print("Start training for <%s> ..." % f_path)
    print("Predict Step: %s" % predict_step)
    x = []
    y = []
    for i in range(4):
        xx, yy = preprocess(data,
                            i + 1,
                            predict_step=predict_step,
                            win=100)

        if len(xx) > 0:
            x.append(xx)
            y.append(yy)
    if not x:
        print("ERROR: invalid data file, len(xx) = 0")
        return
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

    log_path = os.path.join(arg.logpath, TRAINING_LOG_FILE)
    csv_logger = CSVLogger(log_path)

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

    model_path = get_model_path_by_file(arg, f_path, predict_step)
    print("Saving trained model as <%s> ..." % model_path)
    model.save(model_path)

    print("Training for <%s> finished. :)\n" % f_path)


def test(arg, data, f_path, predict_step):
    print("Start testing for <%s> ..." % f_path)
    model_path_1 = get_model_path_by_file(arg, f_path, 1)
    model_path_2 = get_model_path_by_file(arg, f_path, 2)
    model1 = load_model(model_path_1)
    model2 = load_model(model_path_2)

    x = []
    y = []
    llbname = []

    # if data.shape[0] >= 100:
    #     print("ERROR: data.shape[0] < 100")
    #     return

    llbname.append(os.path.basename(f_path))
    lastinput = []
    predict = []
    for i in range(4):
        xx, xlast = preprocess(data,
                               i + 1,
                               predict_step=predict_step,
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

    result_path = os.path.join(arg.resultpath, OUTPUT_FILE)
    print("Appending result to <%s> ..." % result_path)
    with open(result_path, 'a') as fh:
        for i, ll in enumerate(result.tolist()):
            items = ','.join(str(xx) for xx in ll)
            line = '{},{}\n'.format(llbname[i], items)
            fh.write(line)

    print("Testing for <%s> finished. :) \n" % f_path)


def draw(arg):
    png_path = os.path.join(arg.modelpath, PNG_FILE)
    print("Start drawing model as <%s>..." % png_path)
    model = make_model(input_dim=100)
    plot_model(model, to_file=png_path, show_shapes=True)
    print("Drawing model finished. :) \n")


# --------------------------------------------------------------------------------
# exec
# --------------------------------------------------------------------------------

def initial(arg):
    for attr in dir(arg):
        if 'path' in attr:
            p = getattr(arg, attr) 
            absp = os.path.abspath(p)
            setattr(arg, attr, absp)
            print("Using %s: %s ..." % (attr, absp))
            if not os.path.exists(absp):
                os.mkdir(absp)
    print("Using echo: %s ..." % arg.echo)


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', dest='cmd', help='train or test')
    parser.add_argument('--predictstep', dest='predictstep', type=int)
    parser.add_argument('--echo', dest='echo', default=10, type=int)

    parser.add_argument('--log', dest='logpath', default='./log')
    parser.add_argument('--data', dest='datapath', default='./data')
    parser.add_argument('--model', dest='modelpath', default='./model')
    parser.add_argument('--result', dest='resultpath', default='./result')

    return parser.parse_args()


def run(arg):
    exec_train_step1 = bool((arg.cmd == 'train' and arg.predictstep == 1) or not arg.cmd)
    exec_train_step2 = bool((arg.cmd == 'train' and arg.predictstep == 2) or not arg.cmd)
    exec_test = bool(arg.cmd == 'test' or not arg.cmd)
    exec_draw = bool(arg.cmd == 'draw')
    initial(arg)

    segment = '\n------------------------------------------------------------------\n'

    if exec_draw:
        print(segment)
        draw(arg)
    for f_path in gen_file_path(arg.datapath):
        data = pd.read_excel(f_path)
        if exec_train_step1:
            print(segment)
            train(arg, data, f_path, 1)
        if exec_train_step2:
            print(segment)
            train(arg, data, f_path, 2)
        if exec_test:
            print(segment)
            test(arg, data, f_path, 1)
    print(">>>>>>> DONE :) <<<<<<<")


if __name__ == '__main__':
    arg = parse_cmd()
    run(arg)
