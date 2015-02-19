from utils import *
from rma_utils import *

class Matrix(object):
    def __init__(self, rdd, filename, m=None, n=None, cache=False, stack_type=1, repnum=1):
        self.matrix_original = rdd
        self.stack_type = stack_type
        self.repnum = repnum

        if cache:
            self.matrix_original.cache()

        if m is None:
            self.m_original, self.n = self.get_dimensions()
        else:
            self.m_original = m
            self.n = n

        if repnum>1:
            if stack_type == 1:
                self.matrix = self.matrix_original.flatMap(lambda row:[row for i in range(repnum)])
                self.m = self.m_original*repnum
            elif stack_type == 2:
                n = self.n
                self.matrix = add_index(self.matrix_original).flatMap(lambda row: [row[0] for i in range(repnum)] if row[1]<self.m_original-n/2 else [row[0]])
                self.m = (self.m_original-self.n/2)*repnum + self.n/2
        else:
            self.matrix = self.matrix_original
            self.m = m

        self.filename = filename
        self.name = filename + '_stack' + str(stack_type) + '_rep' + str(self.repnum)

    def zip_with_index(self): 
        self.matrix_with_index = add_index(self.matrix)

    def get_dimensions(self):
        m = self.matrix.count()
        try:
            n = len(self.matrix.first())
        except:
            n = 1
        return m, n

    def transpose(self):
        rows_each_part = self.matrix.mapPartitions(lambda x: num_rows_each_partition(x)).collect()
        indexed_matrix = self.matrix.mapPartitionsWithSplit(lambda index, it: indexing(index, it, rows_each_part)). \
            flatMap(lambda x: flip(x))
        grouped = indexed_matrix.groupBy(lambda x: x[0]).map(lambda x: [e for e in x[1]])
        return Matrix(grouped.map(lambda x: extract(x)))

    def dot(self, other):
        # join by columns and sum the corresponding eles
        return Matrix(self.matrix.join(other.matrix).map(lambda (k, v): v). \
                      mapPartitions(other_iterator).sum())

    def take(self, num_rows):
        return self.matrix.take(num_rows)

    def top(self):
        return self.matrix.first()

    def collect(self):
        return self.matrix.collect()
