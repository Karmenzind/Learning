# coding: utf-8

from copy import deepcopy

target = "abcdef"


def perm(idx=0, cur=None):
    if idx == len(target):
        return cur
    if idx == 0:
        cur = [[target[idx]]]
        return perm(idx + 1, cur)
    cur_res = []
    for i in cur:
        for j in range(idx + 1):
            sub_res = deepcopy(i)
            sub_res.insert(j, target[idx])
            cur_res.append(sub_res)
    return perm(idx + 1, cur_res)


if __name__ == '__main__':
    res = sorted(map(lambda x: ''.join(x), perm()))
    print(res)
    print len(res)


