# coding: utf-8

from Hw.common.table import Table

class ReconstructResult(object):

    def __init__(self):
        self.error_no = int() # 0， 1， 2
        self.total_lines = int()
        self.ignored_lines = int()
        self.illegal_lines = int()

        self.table = Table()