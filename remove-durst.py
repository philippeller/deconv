import imageio
import sys
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import zoom
from scipy.ndimage import geometric_transform
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import map_coordinates
from scipy import stats
from mad import median_absolute_deviation


class duster(object):
    def __init__(self, filename, n_frames=None):
        self.reader = imageio.get_reader(filename,  'ffmpeg')
        self.fps = self.reader.get_meta_data()['fps']
        vid_shape = self.reader.get_data(0).shape[:2]
        self.vid_shape = np.array(vid_shape)
        self.center = self.vid_shape/2
        if n_frames:
            assert(n_frames < self.reader.get_length())
            self.n_frames = n_frames
        else:
            self.n_frames = self.reader.get_length() - 1

    def remove_dust(self):
        self.seq = []
        for i in xrange(0,self.n_frames):
            # greyscale conversion
            img = np.average(self.reader.get_data(i),axis=2)
            self.seq.append(img)
        self.seq = np.array(self.seq)
        #var = np.var(self.seq, axis=0)
        #min = np.min(self.seq, axis=0)
        #max = np.max(self.seq, axis=0)
        #delta = max - min
        #var = stats.variation(self.seq, axis=0)
        #gmean = stats.gmean(self.seq, axis=0)
        a = np.average(self.seq, axis=0)
        #grad = ndimage.gaussian_gradient_magnitude(a , 0.25)
        #map = ndimage.prewitt(a)
        map = ndimage.gaussian_laplace(a,2.5) * ndimage.gaussian_gradient_magnitude(a , 0.25)
        cutoff = np.percentile(map,99.9)
        map[map<cutoff]=0
        map[map>0]=1
        #map = grad
        #map[map>300]=300
        fig = plt.figure(figsize=(20,8), frameon=False)
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(map,interpolation='nearest')
        ax1.set_title('variance')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.seq[0], cmap='Greys_r',interpolation='nearest')
        ax2.set_title('img')
        fig.set_tight_layout(True)
        plt.show()



if __name__ == '__main__':

    filename = 'moon_close.avi'
    enhancer = duster(filename)
    enhancer.remove_dust()
