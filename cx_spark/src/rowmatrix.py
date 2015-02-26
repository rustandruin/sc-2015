from utils import *
from rma_utils import *

class rowMatrix(object):
    def __init__(self, rdd, name, m=None, n=None, cache=True, feats=None, stack_type=1, repnum=1):
        self.rdd_original = rdd
        self.stack_type = stack_type
        self.repnum = repnum

        self.name = name
        self.rep_name = name + '_stack' + str(stack_type) + '_rep' + str(self.repnum)

        if m is None:
            self.m_original, self.n = self.get_dimensions()
        else:
            self.m_original = m
            self.n = n

        if feats is None:
            self.feats = range(n)
        else:
            self.feats = feats

        if repnum>1:
            if stack_type == 1:
                self.rdd = self.rdd_original.flatMap(lambda row:[row for i in range(repnum)])
                self.m = self.m_original*repnum
            elif stack_type == 2:
                n = self.n
                self.rdd = add_index(self.rdd_original).flatMap(lambda row: [row[0] for i in range(repnum)] if row[1]<self.m_original-n/2 else [row[0]])
                self.m = (self.m_original-self.n/2)*repnum + self.n/2
        else:
            self.rdd = self.rdd_original
            self.m = m

        self.rdd = convert_rdd(self.rdd)
        self.rdd = add_index(self.rdd)

        if cache:
            self.rdd.cache()
            print 'number of rows: {0}'.format( self.rdd.count() ) #materialize the matrix

    def atamat(self,mat,sc):
        # TO-DO: check dimension compatibility

        mat = sc.broadcast(mat)
        b = self.rdd.mapValues( lambda row: np.dot( np.array(row), mat.value ) )
        b = self.rdd.zip(b).map(lambda ((k1,r1),(k2,r2)): (r1,r2) ).mapPartitions(sumIteratorOuter).sum()
    
        return b

    def rtimes(self,mat,sc,return_rdd=False):
        # TO-DO: check dimension compatibility
        mat = sc.broadcast(mat)
        b = self.rdd.mapValues( lambda row: np.dot( np.array(row), mat.value ) )

        if not return_rdd:
            b = b.collect()
            b.sort(key=lambda x:x[0])
            b = np.array( [ x[1] for x in b ] )

        return b

    def ltimes(self,mat,sc):
        # TO-DO: check dimension compatibility

        mat = sc.broadcast(mat)
        b = self.rdd.map(lambda (key, row): (mat.value[:,key], row)).mapPartitions(sumIteratorOuter).sum()

        return b

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
