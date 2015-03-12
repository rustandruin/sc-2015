#https://github.com/jey/sc-2015/blob/master/spark_msi/spark_msi.py
from lxml import etree
import numpy as np
import scipy as sp
from scipy import sparse
import sys, time, os, operator
import cPickle as pickle

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

def global_mz_axis(spectrum_list, ppm=5):
    """
    Based on the list of dict of all spectra (see read_spectra) compute a global
    mz axis.

    :param sepectrum_list: The output of the read_spectrum function.
    :param ppm: Parts per million
    """
    num_spectra = len(spectrum_list)
    mz_mins = np.zeros(num_spectra)
    mz_maxs = np.zeros(num_spectra)
    for i in range(num_spectra):
        mz_mins[i], mz_maxs[i] = spectrum_list[i]['mz']['range']
    min_mz = mz_mins.min()
    max_mz = mz_maxs.max()
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


class MSIMatrix(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.shape

        def to_matrix(spectrum):
            x, y, t, ions = spectrum
            r = self.to_row(x, y)
            for bucket, mz, intensity in ions:
                c = self.to_col(t, bucket)
                yield (r, c, intensity)

        self.nonzeros = dataset.spectra.flatMap(to_matrix)

    def __getstate__(self):
        # don't pickle RDDs
        result = self.__dict__.copy()
        result['nonzeros'] = None
        return result

    def to_row(self, x, y):
        return self.shape[0]*y + x

    def to_col(self, t, mz_idx):
        return self.shape[2]*mz_idx + t

    def cache(self):
        self.nonzeros.cache()
        return self


class MSIDataset(object):
    # each entry of spectra is (x, y, t, mz_values, intensity_values)
    def __init__(self, mz_axis, spectra, shape=None):
        self.mz_axis = mz_axis
        self.spectra = spectra
        self._shape = shape

    def __getstate__(self):
        # don't pickle RDDs
        result = self.__dict__.copy()
        result['spectra'] = None
        return result

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
        return MSIDataset(self.mz_axis, self.spectra.filter(f), self._shape)

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
        return MSIDataset(self.mz_axis, filtered, self._shape)

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
        metadata = { 'mz_axis' : self.mz_axis, 'shape' : self.shape }
        with file(path + ".meta", 'w') as outf:
            pickle.dump(metadata, outf)

    @staticmethod
    def load(sc, path):
        metadata = pickle.load(file(path + ".meta"))
        spectra = sc.pickleFile(path + ".spectra")
        return MSIDataset(metadata['mz_axis'], spectra, metadata['shape'])

    @staticmethod
    def from_imzml(sc, imzMLPath, imzBinPath):
        # Read all spectra from the XML file
        imzXML = etree.iterparse(imzMLPath, tag='{http://psi.hupo.org/ms/mzml}spectrum')
        spectra_xml = read_spectra(imzXML)

        # Convert the XML data to python dicts with info about the mz and intensity arrays
        spectra = [read_spectrum(i) for i in spectra_xml]
        num_spectra = len(spectra)

        # Compute mz axis
        mz_axis = global_mz_axis(spectra, 5)
        mz_axis_b = sc.broadcast(mz_axis)

        mz_dtype = spectra[0]['mz']['dtype']
        intensity_dtype = spectra[0]['intensity']['dtype']

        def load_spectra(partition):
            imzBin = open(imzBinPath, 'rb')
            for spectrum in partition:
                x, y, t = spectrum['position']
                # skip t==0 because it's just the sum of t=1 to t=200
                if t == 0:
                    continue
                assert spectrum['mz']['dtype'] == mz_dtype
                assert spectrum['intensity']['dtype'] == intensity_dtype
                mz_data, intensity_data = read_raw_data(spectrum, imzBin)
                assert len(mz_data) == len(intensity_data)  # is this a valid asumption?
                mz_bins = [closest_index(mz_axis_b.value, mz) for mz in mz_data]
                assert 1 <= x and 1 <= y and 1 <= t <= 200
                yield (x - 1, y - 1, t - 1, zip(mz_bins, mz_data, intensity_data))

        num_partitions = 32  # FIXME: magic number
        spectra_rdd = sc.parallelize(spectra, num_partitions).mapPartitions(load_spectra)
        return MSIDataset(mz_axis, spectra_rdd).cache()

def converter(sc, imzXMLPath, imzBinPath, outpath):
    return MSIDataset.from_imzml(sc, imzXMLPath, imzBinPath).cache()
