import numpy as np
import sys, time, os, operator, csv
import cPickle as pickle
from bisect import bisect_left

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)

def get_all_items(element, itemlist=None):
    """
    Get all items for all children and recursive subchildren for an element, e.g., a spectrum
    """
    if itemlist is None:
        itemlist = []
    itemlist.append(dict(element.items())) # make dict
    for e in element:
        itemlist.append(dict(e.items())) # make dict
        for ec in e:
            get_all_items(ec, itemlist)
    return itemlist

def read_spectrum(xml_element):
    """
    Read the info about the spectrum from xml
    """
    spectrum_info = get_all_items(xml_element)
    #if not spectrum_info[1]['name'].startswith('MS1'):
    #    print spectrum_info[1]['name']
    position = (int(spectrum_info[8]['value']),
                int(spectrum_info[9]['value']),
                int(spectrum_info[10]['value']))
    index = int(spectrum_info[0]['index'])
    mz = {'range': (float(spectrum_info[3]['value']),float(spectrum_info[4]['value'])),
          'num_values': int(spectrum_info[14]['value']),
          'offset': int(spectrum_info[15]['value']),
          'length': int(spectrum_info[16]['value']),
          'dtype': 'float32'} # TODO: This should be retrieved from the header
    intensity = {'num_values': int(spectrum_info[20]['value']),
                 'offset': int(spectrum_info[21]['value']),
                 'length': int(spectrum_info[22]['value']),
                 'dtype': 'float32'} # TODO: This should be retrieved from the header
    return {'position': position,
            'index': index,
            'mz': mz,
            'intensity': intensity}

def read_spectra(xml_file):
    return [element for event, element in xml_file]

@memoize
def get_mz_axis(mz_range, ppm=5):
    min_mz, max_mz = mz_range
    f = np.ceil(1e6*np.log(max_mz/min_mz)/ppm)
    mz_edges = np.logspace(np.log10(min_mz),np.log10(max_mz),f)
    return mz_edges

def read_raw_data(spectrum_dict, data_file):
    """
    Take the output of read_sepctrum and read the raw data for the mz and intensity array.

    :param spectrum_dict: Output of read_spectrum, i.e., a dict describing the spectrum.
    :param data_file: The binary file with the raw data.

    """
    data_file.seek(spectrum_dict['mz']['offset'])
    mz_data = np.fromfile(data_file,
                          dtype=spectrum_dict['mz']['dtype'],
                          count=spectrum_dict['mz']['num_values'],
                          sep="")
    data_file.seek(spectrum_dict['intensity']['offset'])
    intensity_data = np.fromfile(data_file,
                                 dtype=spectrum_dict['intensity']['dtype'],
                                 count=spectrum_dict['intensity']['num_values'],
                                 sep="")
    return mz_data, intensity_data


# binary search used to lookup mz index
# TODO: move this function
def closest_index(data, val):
    def closest(highIndex, lowIndex):
      if abs(data[highIndex] - val) < abs(data[lowIndex] - val):
        return highIndex
      else:
        return lowIndex
    highIndex = len(data)-1
    lowIndex = 0
    while highIndex > lowIndex:
       index = (highIndex + lowIndex) / 2
       sub = data[index]
       if data[lowIndex] == val:
           return lowIndex
       elif sub == val:
           return index
       elif data[highIndex] == val:
           return highIndex
       elif sub > val:
           if highIndex == index:
             return closest(highIndex, lowIndex)
           highIndex = index
       else:
           if lowIndex == index:
             return closest(highIndex, lowIndex)
           lowIndex = index
    return closest(highIndex, lowIndex)


# Note: "raw" means with empty rows/cols
class MSIMatrix(object):
    def __init__(self, dataset_shape, raw_shape, shape, seen_bcast, nonzeros):
        self.dataset_shape = dataset_shape
        self.raw_shape = raw_shape
        self.shape = shape
        self.seen_bcast = seen_bcast
        self.nonzeros = nonzeros

    @staticmethod
    def from_dataset(sc, dataset):
        xlen, ylen, tlen, mzlen = dataset.shape
        raw_shape = (xlen * ylen, tlen * mzlen)

        def to_raw_matrix(spectrum):
            x, y, t, ions = spectrum
            r = y * xlen + x
            for bucket, mz, intensity in ions:
                c = bucket * tlen + t
                yield (r, c, intensity)

        raw_nonzeros = dataset.spectra.flatMap(to_raw_matrix)
        raw_nonzeros = raw_nonzeros.repartition(8192)  # FIXME HACK?
        raw_nonzeros = raw_nonzeros.cache()
        seen_rows = sorted(raw_nonzeros.map(lambda (r, c, v): r).mapPartitions(set).distinct().collect())
        seen_cols = sorted(raw_nonzeros.map(lambda (r, c, v): c).mapPartitions(set).distinct().collect())
        seen_bcast = sc.broadcast((seen_rows, seen_cols))
        shape = (len(seen_rows), len(seen_cols))

        def to_matrix(entry):
            rowids, colids = seen_bcast.value
            row, col, value = entry
            newrow = bisect_left(rowids, row)
            assert newrow != len(rowids)
            newcol = bisect_left(colids, col)
            assert newcol != len(colids)
            return (newrow, newcol, value)

        nonzeros = raw_nonzeros.map(to_matrix)
        return MSIMatrix(dataset.shape, raw_shape, shape, seen_bcast, nonzeros)

    def __getstate__(self):
        # don't pickle RDDs
        result = self.__dict__.copy()
        result['nonzeros'] = None
        return result

    def index(x, y, t, mz_idx):
        return (self.row(x, y), self.col(t, mz_idx))

    def row(self, x, y):
        rowids, colids = self.seen_bcast.value
        row = bisect_left(rowids, self.raw_row(x, y))
        assert row != len(rowids)
        return row

    def col(self, t, mz_idx):
        rowids, colids = self.seen_bcast.value
        col = bisect_left(colids, self.raw_col(t, mz_idx))
        assert col != len(colids)
        return col

    def raw_row(self, x, y):
        return self.dataset_shape[0]*y + x

    def raw_col(self, t, mz_idx):
        return self.dataset_shape[2]*mz_idx + t

    def cache(self):
        self.nonzeros.cache()
        return self

    def save(self, path):
        self.nonzeros.map(lambda entry: ",".join(map(str, entry))).saveAsTextFile(path + ".csv")
        metadata = {
            'dataset_shape' : self.dataset_shape,
            'raw_shape' : self.raw_shape,
            'shape' : self.shape,
            'seen' : self.seen_bcast.value
        }
        with file(path + ".meta", 'w') as outf:
            pickle.dump(metadata, outf)

    @staticmethod
    def load(sc, metapath, csvpath):
        with file(metapath) as inf:
            meta = pickle.load(inf)
        def parse_nonzero(line):
            row, col, value = line.split(',')
            return (int(row), int(col), float(value))
        nonzeros = sc.textFile(csvpath).map(parse_nonzero)
        seen_bcast = sc.broadcast(meta['seen'])
        result = MSIMatrix(
                meta['dataset_shape'],
                meta['raw_shape'],
                meta['shape'],
                seen_bcast,
                nonzeros)
        return result


class MSIDataset(object):
    # each entry of spectra is (x, y, t, mz_values, intensity_values)
    def __init__(self, mz_range, spectra, shape=None, mask=None):
        self.spectra = spectra
        self.mz_range = mz_range
        self._shape = shape
        self.mask = mask

    def apply_mask(self, mask):
        print self.shape, mask.shape
        assert mask.dtype == np.bool_
        assert mask.shape == (self.shape[1] + 1, self.shape[0] + 1)
        assert not mask[:, -1].any()
        assert not mask[-1, :].any()
        def is_not_masked(spectrum):
            x, y, t, ions = spectrum
            return mask[y, x]
        spectra = self.spectra.filter(is_not_masked)
        if self.mask is not None:
            mask = np.logical_and(self.mask, mask)
        return MSIDataset(self.mz_range, self.spectra, self._shape, mask)

    def __getstate__(self):
        # don't pickle RDDs
        result = self.__dict__.copy()
        result['spectra'] = None
        return result

    @property
    def mz_axis(self):
        return get_mz_axis(self.mz_range)

    @property
    def shape(self):
        if self._shape is None:
            mz_len = len(self.mz_axis)
            def mapper(spectrum):
                x, y, t, ions = spectrum
                for bucket, mz, inten in ions:
                    assert 0 <= bucket < mz_len
                return (x, y, t)
            def reducer(a, b):
                return map(max, zip(a, b))
            extents = self.spectra.map(mapper).reduce(reducer)
            self._shape = (extents[0] + 1, extents[1] + 1, extents[2] + 1, mz_len)
        return self._shape

    # Filter to only keep a sub-array of the cube
    def select(self, xs, ys, ts):
        def f(spectrum):
            x, y, t, ions = spectrum
            return x in xs and y in ys and t in ts
        # TODO: compute actual shape
        return MSIDataset(self.mz_range, self.spectra.filter(f), self._shape, self.mask)

    def __getitem__(self, key):
        def mkrange(arg, hi):
            if isinstance(arg, slice):
                if arg.step is not None and arg.step != 1:
                    raise NotImplementedError("step != 1")
                start = arg.start if arg.start is not None else 0
                stop = arg.stop if arg.stop is not None else hi + 1
                return xrange(start, stop)
            else:
                return xrange(arg, arg + 1)
        xs, ys, ts = key
        xs = mkrange(xs, self.shape[0])
        ys = mkrange(ys, self.shape[1])
        ts = mkrange(ts, self.shape[2])
        return self.select(xs, ys, ts)

    def select_mz_range(self, lo, hi):
        def f(spectrum):
            x, y, t, ions = spectrum
            new_ions = filter(lambda (bucket, mz, inten): lo <= mz <= hi, ions)
            if len(new_ions) > 0:
                yield (x, y, t, new_ions)
        filtered = self.spectra.flatMap(f)
        return MSIDataset(self.mz_range, filtered, self._shape, self.mask)

    # Returns sum of intensities for each mz
    def histogram(self):
        def f(spectrum):
            x, y, t, ions = spectrum
            for bucket, mz, intensity in ions:
                yield mz, intensity
        results = self.spectra.flatMap(f).reduceByKey(operator.add).sortByKey().collect()
        return zip(*results)

    def intensity_vs_time(self):
        def f(spectrum):
            x, y, t, ions = spectrum
            return (t, sum([intensity for (bucket, mz, intensity) in ions]))
        results = self.spectra.map(f).reduceByKey(operator.add).sortByKey().collect()
        return zip(*results)

    def image(self):
        def f(spectrum):
            x, y, t, ions = spectrum
            return ((x, y), sum([inten for (bucket, mz, inten) in ions]))
        pixels = self.spectra.map(f).reduceByKey(operator.add).collect()
        result = np.zeros((self.shape[0], self.shape[1]))
        for (x, y), inten in pixels:
            result[x, y] = inten
        return result

    def cache(self):
        self.spectra.cache()
        return self

    def save(self, path):
        self.spectra.saveAsPickleFile(path + ".spectra")
        metadata = { 'mz_range' : self.mz_range, 'shape' : self.shape }
        with file(path + ".meta", 'w') as outf:
            pickle.dump(metadata, outf)

    @staticmethod
    def load(sc, metapath, rddpath, minPartitions=None):
        metadata = pickle.load(file(metapath + ".meta"))
        spectra = sc.pickleFile(rddpath + ".spectra", minPartitions=minPartitions)
        return MSIDataset(metadata['mz_range'], spectra, metadata['shape'], metadata.get('mask', None))

    @staticmethod
    def dump_imzml(imzMLPath, outpath, chunksz=10**5):
        def genrow():
            # yields rows of form [x, y, t, num_values, mz_offset, intensity_offset]
            from lxml import etree
            imzXML = etree.iterparse(imzMLPath, tag='{http://psi.hupo.org/ms/mzml}spectrum')
            for event, element in imzXML:
                if event == 'end':
                    spectrum = read_spectrum(element)
                    mz = spectrum['mz']
                    intensity = spectrum['intensity']
                    assert mz['num_values'] == intensity['num_values']
                    assert mz['length'] == 4 * mz['num_values']
                    assert intensity['length'] == 4 * intensity['num_values']
                    yield list(spectrum['position']) + [mz['num_values'], mz['offset'], intensity['offset']]
                element.clear()

        def genfilename():
            i = 0
            while True:
                i += 1
                print >> sys.stderr, i
                yield os.path.join(outpath, "%05d" % i)

        os.mkdir(outpath)
        rows = genrow()
        filenames = genfilename()
        row = None
        done = False
        while not done:
            with open(next(filenames), 'w') as outf:
                out = csv.writer(outf)
                for line_num in xrange(chunksz):
                    try:
                        row = next(rows)
                    except StopIteration:
                        done = True
                        break
                    out.writerow(row)

    @staticmethod
    def from_dump(sc, path, imz_path):
        def load_spectrum(imz_file, mz_offset, intensity_offset, num_values):
            imz_file.seek(mz_offset)
            mz_data = np.fromfile(
                            imz_file,
                            dtype='float32',
                            count=num_values,
                            sep="")
            imz_file.seek(intensity_offset)
            intensity_data = np.fromfile(
                                    imz_file,
                                    dtype='float32',
                                    count=num_values,
                                    sep="")
            return mz_data, intensity_data


        def load_part(partition):
            with open(imz_path, 'rb') as imz_file:
                for row in partition:
                    x, y, t, num_values, mz_offset, intensity_offset = row
                    assert 1 <= x and 1 <= y and 0 <= t <= 200
                    # skip t==0 because it's just the sum of t=1 to t=200
                    if t == 0:
                        continue
                    mz_data, intensity_data = load_spectrum(imz_file, mz_offset, intensity_offset, num_values)
                    yield (x - 1, y - 1, t - 1, zip(mz_data, intensity_data))

        # load the spectra (unbinned)
        num_partitions = 1024 # minimum parallelism
        spectra_rdd = sc.textFile(path, num_partitions).map(lambda row: [int(v) for v in row.split(',')])
        #spectra_rdd = spectra_rdd.sortBy(lambda row: row[4])
        spectra_rdd = spectra_rdd.mapPartitions(load_part)
        spectra_rdd.cache()

        # compute min and max mz
        def minmax_map(spectrum):
            x, y, t, ions = spectrum
            mz_data, intensity_data = zip(*ions)
            return (min(mz_data), max(mz_data))
        def minmax_reduce(a, b):
            lo = min(a[0], b[0])
            hi = max(a[1], b[1])
            return (lo, hi)
        mz_range = spectra_rdd.map(minmax_map).reduce(minmax_reduce)

        # compute mz buckets
        def apply_buckets(partition):
            mz_axis = get_mz_axis(mz_range)
            for spectrum in partition:
                x, y, t, ions = spectrum
                mz_data, intensity_data = zip(*ions)
                mz_buckets = [closest_index(mz_axis, mz) for mz in mz_data]
                new_ions = zip(mz_buckets, mz_data, intensity_data)
                yield x, y, t, new_ions
        spectra_rdd = spectra_rdd.mapPartitions(apply_buckets)
        spectra_rdd.unpersist()
        return MSIDataset(mz_range, spectra_rdd).cache()


def converter():
    imzXMLPath = 'Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.imzml'
    imzBinPath = "Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.ibd"
    csvpath = "Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.csv"
    outpath = "Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.rdd"
    MSIDataset.dump_imzml(imzXMLPath, csvpath)
    from pyspark import SparkContext
    sc = SparkContext()
    MSIDataset.from_dump(sc, csvpath, imzBinPath).save(outpath)


if __name__ == '__main__':
    #prefix = 's3n://AKIAIKCSJKIZFWQBNTCQ:Is0OfXC8%2FF33sJRMF7axXXtjeKwXoyn65aL01OGt@amp-jey/'
    #prefix = 's3n://amp-jey-west/'
    prefix = 'hdfs:///sc-2015/'
    metapath = 'Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1.rdd'
    rddpath = prefix + 'sc-2015/Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.rdd'
    matpath = prefix + 'sc-2015/Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat'
    from pyspark import SparkContext
    sc = SparkContext()
    dataset = MSIDataset.load(sc, metapath, rddpath)
    with open('output.pkl', 'w') as outf:
        pickle.dump(dataset.image(), outf)


if __name__ == '__main2__':
    # big
    name = 'Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked'
    # small
    #name = '2014Nov15_PDX_IMS_imzML'
    datapath = '/scratch1/scratchdirs/jeyk/data/'
    inpath = os.path.join(datapath, name, name)
    outpath = os.path.join(os.getenv('SCRATCH'), 'dev-data', name, name)
    from pyspark import SparkContext
    sc = SparkContext()
    dataset = MSIDataset.load(sc, inpath + ".rdd")
    dataset = dataset[100:200, 100:200, :]
    mat = MSIMatrix(dataset)
    print "shape: ", mat.raw_shape, " -> ", mat.shape
    mat.save(outpath + "-test.mat")
    sc.stop()
