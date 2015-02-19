import numpy as np
import cPickle as pickle
import json

def parseVector(line):
    return np.array([float(x) for x in line.strip('\n').split(' ')])

def add(x, y):
    x += y
    return x

def sampling(row, R, sumLev, s):
    #row = row.getA1()
    lev = np.linalg.norm(np.dot(row[:-1], R))**2
    p = s*lev/sumLev
    coin = np.random.rand()
    if coin < p:
        return row/p

def unifSampling(row, n, s):
    #row = row.getA1()
    p = s/n
    coin = np.random.rand()
    if coin < p:
        return row/p

def pickle_load(filename):
    return pickle.load(open( filename, 'rb' ))

def pickle_write(filename,data):
    with open(filename, 'w') as outfile:
        pickle.dump(data, outfile, True)

def json_write(filename,*args):
    with open(filename, 'w') as outfile:
        for data in args:
            json.dump(data, outfile)
            outfile.write('\n')

class Block_Mapper:
    """
    process data after receiving a block of records
    """
    def __init__(self, blk_sz=50):
        self.blk_sz = blk_sz
        self.data = []
        self.sz = 0
        self.key = []
    
    def __call__(self, records):
        for row in records:
            a = self.parse(row)
            if len(a) == 2:
                self.key.append(a[0])
                self.data.append(a[1])
            else:
                self.data.append(a)
            self.sz += 1
                
            if self.sz >= self.blk_sz:
                for key, value in self.process():
                    yield key, value
                self.data = []
                self.key = []
                self.sz = 0
        if self.sz > 0:
            for key, value in self.process():
                yield key, value
        for key, value in self.close():
            yield key, value

    def parse(self, row):
        return row

    def process(self):
        return iter([])
    
    def close(self):
        return iter([])

  
