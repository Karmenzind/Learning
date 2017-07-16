# coding: utf-8

class Table(object):
    def __init__(self):
        self.columns = []

    def add_column(self, column):
        self.columns.append(column)

    def get_columns(self):
        return self.columns