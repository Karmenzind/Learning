# !/usr/bin/env/python
# -*- coding: utf-8 -*-

import re

from Hw.common.reconstruct_result import ReconstructResult
from Hw.common.column import Column
from Hw.common.table_reconstruct import TableReconstruct


class TableReconstructor(object):
    def __init__(self):
        self.res = {"error_no": 0, "total_lines": 0, "ignored_lines": 0, "illegal_lines": 0}
        self.index = self.count = self.sort = ''
        self.meta_rows = ''
        self.name_row = self.type_row = self.default_row = ''
        self.col_name_lst = []
        self.type_lst = []
        self.col_num = 0

        self.type_map = {"": lambda x: True,
                         'string': lambda x: True,
                         'int': lambda x: True if re.match('\d+', x) else False,
                         'long': lambda x: True if re.match('\d+', x) else False,
                         'float': lambda x: True if re.match('\d+[.]\d+', x) else False,
                         'boolean': lambda x: True if re.match('true|false', x) else False}

    def do_reconstruct(self, index, count, sort=0, data=',Male', *args, **kw):
        """
        :param index: 源列索引 
        :param count: 新增列集输出数量上限
        :param sort: 0,列名升序；1，枚举值频度升序
        :param data:  # "1", 2, 0, data
        :param args: 
        :param kw: 
        :return: 
        """

        self.count = count
        self.sort = sort
        self.rows = data.strip().rstrip(';').split(';')
        self.col_name_lst = self.rows[0].split(',')
        self.col_num = len(self.col_name_lst)
        result = ReconstructResult()
        self.meta_rows = self.rows[:3]  # 元数据行
        self.data_rows = self.rows[3:]  # 数据行

        try:
            assert isinstance(index, basestring)
            self.index = map(int, index.split(','))

        except (ValueError, AssertionError):
            self.res['error_no'] = 1

        self.index_judge()
        self.meta_rows_judge()

        if self.res['error_no'] == 0:
            for x in [col_obj for col_obj in self.handle_data(data)]:
                result.table.add_column(x)
            result.illegal_lines = self.res['illegal_lines']
            result.total_lines = self.res['total_lines']
            result.ignored_lines = self.res['ignored_lines']
            if result.total_lines > result.ignored_lines + result.illegal_lines:
                result.list = result.table
                self._sort(result)
        return result

    def _sort(self, result):
        src_idx = [0, self.index[0]][self.sort]
        src_key = result.table.columns[src_idx].values
        sorted_key = sorted(src_key, reverse=True)
        sorted_idx = map(lambda x: src_key.index(x), sorted_key)
        for idx, column in enumerate(result.list.columns):
            init_values = column.values
            result.list.columns[idx].values = map(lambda x: init_values[x], sorted_idx)
        result.list.columns = sorted(result.table.columns, key=lambda x: x.column_name, reverse=True)

    def handle_data(self, data):

        self.res['total_lines'] = len(self.data_rows)
        illegal_lst = []
        append_res = []
        for idx, col_name in enumerate(self.col_name_lst):
            tmp_column_obj = Column(col_name, [])
            for row_idx, row in enumerate(self.data_rows):
                val_list = row.split(',')
                tmp_column_obj.add_values(val_list[idx])
                for val in val_list:
                    type_func = self.type_map[self.type_lst[idx]]
                    if not type_func(val):
                        illegal_lst.append(row)
            # 新增列
            yield tmp_column_obj
            if idx in self.index:
                enums = set(tmp_column_obj.values)
                for enum in enums:
                    append_name = "Flag_{}_{}".format(self.col_name_lst[idx], enum)
                    append_list = []
                    for val in tmp_column_obj.values:
                        append_val = [False, True][val == enum]
                        append_list.append(append_val)
                    append_col_obj = Column(append_name, append_list)
                    append_res.append(append_col_obj)
        for append_obj in append_res:
            yield append_obj
        self.res['total_lines'] = len(self.data_rows)
        self.res['illegal_lines'] = len(illegal_lst)

    def meta_rows_judge(self):
        if len(self.meta_rows) < 3:
            self.res['error_no'] = 2
        else:
            self.name_row, self.type_row, self.default_row = self.meta_rows
            self.type_lst = self.type_row.split(',')
            for _type in self.type_lst:
                if _type.lower() not in ('', 'string', 'int', 'long', 'float', 'boolean'):
                    self.res['error_no'] = 2
            if not len(self.name_row.split(',')) == len(self.type_lst) == len(self.default_row.split(',')):
                self.res['error_no'] = 2

    def data_row_judge(self, col_num, type_row, val_list):
        if not self.data_rows:
            self.res['error_no'] = 2
        # 数据类型
        # 列数

    def index_judge(self):
        for _idx in self.index:
            if _idx < 0 or _idx > self.col_num - 1:
                self.res['error_no'] = 1

data = 'Name,Gender;,;,Male;a,Male;b,Female'
a = TableReconstructor().do_reconstruct("1", 2, 0, data)

print a.table.get_columns()

#
# import unittest
#
#
# class Test(unittest.TestCase):
#     def testcase01(self):
#         tableR = TableReconstructor()
#         data = "Name,Gender;,;,Male;a,Male;b,Female"
#         result = tableR.do_reconstruct("1", 2, 0, data)
#
#         rtn = TableReconstruct.result_2_str(result)
#         # self.assertEqual()
