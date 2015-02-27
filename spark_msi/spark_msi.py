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

def global_axes(spectrum_list, ppm=5, continuous_time_axis=True):
    """
    Based on the list of dict of all spectra (see read_spectra) compute a global
    x axis , y axis, time axis, and mz_axis.

    :param sepectrum_list: The output of the read_spectrum function.
    :param continuous_time_axis=: Boolean indicating whether we need a continuous axis.

    :returns: dict with 4 numpy arrays, one per axis
    """
    temp = np.asarray([i['position'] for i in  spectrum_list])
    min_vals = temp.min(axis=0)
    max_vals = temp.max(axis=0)
    x_axis = np.arange(min_vals[0], max_vals[0]+1, 1) - min_vals[0]
    y_axis = np.arange(min_vals[1], max_vals[1]+1, 1) - min_vals[1]
    if not continuous_time_axis:
        time_axis = np.unique(np.asarray([i['position'][2] for i in  spectrum_list]) , False, False)
    else:
        time_axis = np.arange(min_vals[2], max_vals[2]+1, 1)
    mz_axis = global_mz_axis(spectrum_list=spectrum_list,
                             ppm=ppm)
    return {'x': x_axis,
            'y': y_axis,
            'mz': mz_axis,
            'time': time_axis}

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


class MSIDataset(object):
    # each entry of spectra is (x, y, t, mz_values, intensity_values)
    def __init__(self, axes, spectra, shape=None):
        self.axes = axes
        self.spectra = spectra
        self._shape = shape

    @property
    def shape(self):
        if self._shape is None:
            def mapper(spectrum):
                x, y, t, ions = spectrum
                return (x, x, y, y, t, t)
            def reducer(a, b):
                ax0, ax1, ay0, ay1, at0, at1 = a
                bx0, bx1, by0, by1, bt0, bt1 = b
                x0 = min(ax0, bx0)
                x1 = max(ax1, bx1)
                y0 = min(ay0, by0)
                y1 = max(ay1, by1)
                t0 = min(at0, bt0)
                t1 = max(at1, bt1)
                return (x0, x1, y0, y1, t0, t1)
            x0, x1, y0, y1, t0, t1 = self.spectra.map(mapper).reduce(reducer)
            self._shape = ((x0, x1), (y0, y1), (t0, t1))
        return self._shape

    @property
    def mz_axis(self):
        return self.axes['mz']

    # Filter to only keep a sub-array of the cube
    def select(self, xs, ys, ts):
        def f(spectrum):
            x, y, t, ions = spectrum
            return x in xs and y in ys and t in ts
        # FIXME: compute actual shape
        return MSIDataset(self.axes, self.spectra.filter(f), None)

    # Filter to keep a single (x, y, t) position
    def select1(self, x0, y0, t0):
        def f(spectrum):
            x, y, t, ions = spectrum
            return x == x0 and y == y0 and t == t0
        # FIXME: compute actual shape
        return MSIDataset(self.axes, self.spectra.filter(f), None)

    def __getitem__(self, key):
        def mkrange(arg, lo, hi):
            if isinstance(arg, slice):
                if arg.step is not None and arg.step != 1:
                    raise NotImplementedError("step != 1")
                start = arg.start if arg.start is not None else lo
                stop = arg.stop if arg.stop is not None else hi + 1
                return xrange(start, stop)
            else:
                return xrange(arg, arg + 1)
        xs, ys, ts = key
        xs = mkrange(xs, *self.shape[0])
        ys = mkrange(ys, *self.shape[1])
        ts = mkrange(ts, *self.shape[2])
        return self.select(xs, ys, ts)

    def select_mz_range(self, lo, hi):
        def f(spectrum):
            x, y, t, ions = spectrum
            new_ions = filter(lambda (bucket, mz, inten): lo <= mz <= hi, ions)
            if len(new_ions) > 0:
                yield (x, y, t, new_ions)
        filtered = self.spectra.flatMap(f)
        return MSIDataset(self.axes, filtered, self.shape)

    # Returns sum of intensities for each mz bin
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
        (min_x, max_x), (min_y, max_y), (min_t, max_t) = self.shape
        result = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
        for (x, y), inten in pixels:
            result[x - min_x, y - min_y] = inten
        return result

    def cache(self):
        self.spectra.cache()
        return self

    def save(self, path):
        self.spectra.saveAsPickleFile(path + ".spectra")
        metadata = { 'axes' : self.axes, 'shape' : self.shape }
        with file(path + ".meta", 'w') as outf:
            pickle.dump(metadata, outf)

    @staticmethod
    def load(sc, path):
        metadata = pickle.load(file(path + ".meta"))
        spectra = sc.pickleFile(path + ".spectra")
        return MSIDataset(metadata['axes'], spectra, metadata['shape'])

    @staticmethod
    def from_imzml(sc, imzMLPath, imzBinPath):
        # Read all spectra from the XML file
        imzXML = etree.iterparse(imzMLPath, tag='{http://psi.hupo.org/ms/mzml}spectrum')
        spectra_xml = read_spectra(imzXML)

        # Convert the XML data to python dicts with info about the mz and intensity arrays
        spectra = [read_spectrum(i) for i in spectra_xml]
        num_spectra = len(spectra)

        # Compute mz axis
        axes = global_axes(spectra, 5, False)

        mz_axis_b = sc.broadcast(axes['mz'])
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
                yield (x, y, t, zip(mz_bins, mz_data, intensity_data))

        num_partitions = 32  # FIXME: magic number
        spectra_rdd = sc.parallelize(spectra, num_partitions).mapPartitions(load_spectra)
        return MSIDataset(axes, spectra_rdd)


def converter(sc):
    imzXMLPath = 'Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.imzml'
    imzBinPath = "Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.ibd"
    outpath = "Lewis_Dalisay_Peltatum_20131115_PDX_Std_1.rdd"
    MSIDataset.from_imzml(sc, imzXMLPath, imzBinPath).save(outpath)


if __name__ == '__main__':
    from pyspark import SparkContext
    converter(SparkContext("local[4]"))
