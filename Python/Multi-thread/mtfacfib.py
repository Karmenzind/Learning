# coding: utf-8

import threading
from time import sleep, ctime


class MyThread(threading.Thread):

    def __init__(self, func, args, name=''):
        super(MyThread, self).__init__()
        self.name = name
        self.args = args
        self.func = func

    def get_result(self):
        return self.res

    def run(self):
        print "starting", self.name, 'at:', ctime()
        self.res = self.func(*self.args)
        print self.name, "finished at:", ctime()


def fib(x):
    sleep(0.005)
    if x < 2:
        return 1
    return fib(x-2) + fib(x-1)


def fac(x):
    sleep(0.1)
    if x < 2:
        return 1
    return x * fac(x-1)


def sum(x):
    sleep(0.1)
    if x < 2:
        return 1
    return x + sum(x-1)


funcs = [fib, fac, sum]
n = 12


def main():
    print '*** SINGLE THREAD'
    for idx, func in enumerate(funcs):
        print 'starting', func.__name__, 'at: ', ctime()
        print func(n)
        print func.__name__, 'finished at:', ctime()
    print '\n*** MULTIPLE THREADS'
    threads = []
    for idx, func in enumerate(funcs):
        t = MyThread(func, (n,), func.__name__)
        threads.append(t)
    for t in threads:
        t.start()

    for t in threads:
        t.join()
        print t.get_result()
    print "all DONE"

if __name__ == '__main__':
    main()
