# coding: utf-8


class Column(object):

    """"""
    def __init__(self, column_name, values):
        self.column_name = column_name
        self.values = values

    def add_values(self, value):
        self.values.append(value)

    def get_column_name(self):
        return self.column_name

    def get_values(self):
        return self.values