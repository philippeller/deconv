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

def trans_worker(params):
    frame, average, patch_size, crop = params

    grid_edges = np.arange(0,2*crop+1,patch_size)

    grid_x, grid_y = np.mgrid[0:2*crop-1:np.complex(2*crop), 0:2*crop-1:np.complex(2*crop)]

    source = []
    destination = []

    xx_out = []
    yy_out = []

    for x in xrange(len(grid_edges)):
        for y in xrange(len(grid_edges)):
            destination.append([grid_edges[x],grid_edges[y]])
            x_0 = grid_edges[x]
            y_0 = grid_edges[y]
            x_m = patch_size/2 if x > 0 else 0
            y_m = patch_size/2 if y > 0 else 0
            x_p = patch_size/2 if x < len(grid_edges)-1 else 0
            y_p = patch_size/2 if y < len(grid_edges)-1 else 0
            x_low = x_0 - x_m
            x_high = x_0 + x_p
            y_low = y_0 - y_m
            y_high = y_0 + y_p
            patch = frame[x_low:x_high,y_low:y_high]
            res = []
            for move_x in xrange(-x_m,x_p):
                for move_y in xrange(-y_m,y_p):
                    ref_x0 = x_low+move_x
                    ref_x1 = x_high+move_x
                    ref_y0 = y_low+move_y
                    ref_y1 = y_high+move_y
                    ref_patch = average[ref_x0:ref_x1,ref_y0:ref_y1]
                    res.append(np.sum(np.square(patch-ref_patch)))
            minimum = np.argmin(res)
            x_min = minimum/(x_m+x_p)-x_m
            y_min = minimum%(y_m+y_p)-y_m
            x_out = grid_edges[x]-x_min
            y_out = grid_edges[y]-y_min
            xx_out.append(x_out)
            yy_out.append(y_out)
            source.append([x_out, y_out])

    source = np.array(source)
    destination = np.array(destination)

    grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')

    trans = []
    for grid_z_p in grid_z:
        trans.append(map_coordinates(frame, grid_z_p.T,mode='reflect'))
    return trans

class CIA_enhance(object):
    def __init__(self, filename, n_frames=None, patch_size=8):
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
        self.crop = 128
        assert(self.crop % patch_size == 0)
        self.patch_size = patch_size

    def align_and_crop(self):
        self.seq = []
        self.alignement_error = [[],[]]
        for i in xrange(0,self.n_frames):
            # greyscale conversion
            img = np.average(self.reader.get_data(i),axis=2)
            for n in range(2):
                com = ndimage.measurements.center_of_mass(img)
                for axis in [0,1]:
                    img = np.roll(img, int(round(self.center[axis]-com[axis])), axis=axis)
                com = np.array(ndimage.measurements.center_of_mass(img))
                self.alignement_error[n].append(np.sum(np.square(com-self.center)))
            # crop
            lower_bounds = self.center - self.crop
            upper_bounds = self.center + self.crop
            img_c = img[lower_bounds[0]:upper_bounds[0],lower_bounds[1]:upper_bounds[1]]
            com = np.array(ndimage.measurements.center_of_mass(img_c))
            self.seq.append(img_c)
        

    def transform(self):
        self.seq = np.array(self.seq)
        average = np.average(self.seq,axis=0)
        #self.seq_trans = []

        average_list = [average]*len(self.seq)

        params = zip(self.seq, average_list, [self.patch_size]*len(self.seq) , [self.crop]*len(self.seq))
        pool = multiprocessing.Pool()
        self.seq_trans = pool.map(trans_worker, params)

        #for frame in self.seq:
        #    trans = trans_worker(frame, average, self.patch_size, self.crop)
        #    self.seq_trans.append(np.array(trans))

        self.seq_trans = np.array(self.seq_trans)
        average_trans = np.average(self.seq_trans,axis=0)
        average_list = np.array(average_list)

        residuals_before = np.mean(np.square(self.seq - average_list),axis=(1,2))
        residuals_after = np.mean(np.square(self.seq_trans - average_list),axis=(1,2))
        central_residuals_before = np.mean(np.square(self.seq - average_list)[:,2*self.patch_size:-2*self.patch_size,2*self.patch_size:-2*self.patch_size],axis=(1,2))
        central_residuals_after = np.mean(np.square(self.seq_trans - average_list)[:,2*self.patch_size:-2*self.patch_size,2*self.patch_size:-2*self.patch_size],axis=(1,2))


        #fig = plt.figure(figsize=(15,15), frameon=False)
        #fig.subplots_adjust(hspace=0)
        #fig.subplots_adjust(wspace=0)
        #ax1 = fig.add_subplot(1, 2, 1)
        #ax1.imshow(average, cmap='Greys_r')
        #ax1.set_xlim(grid_edges[0],grid_edges[-1])
        #ax1.set_ylim(grid_edges[0],grid_edges[-1])
        #ax1.set_title('Average')

        #ax2 = fig.add_subplot(1, 2, 2)
        #ax2.imshow(average_trans, cmap='Greys_r')
        #ax2.set_xlim(grid_edges[0],grid_edges[-1])
        #ax2.set_ylim(grid_edges[0],grid_edges[-1])
        #ax2.set_title('Average Transformed')
        #writer = imageio.get_writer('average.jpeg',quality=100)
        #writer.append_data(average)
        #writer.close()
        #writer = imageio.get_writer('average.tif')
        #writer.append_data(average)
        #writer.close()
        #writer = imageio.get_writer('taverage.jpeg',quality=100)
        #writer.append_data(average_trans)
        #writer.close()
        #writer = imageio.get_writer('taverage.tif')
        #writer.append_data(average_trans)
        #writer.close()

        i = 3

        fig = plt.figure(figsize=(20,12), frameon=False)
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(average, cmap='Greys_r')
        ax1.set_title('Average')
        #ax1.contour(average)

        ax6 = fig.add_subplot(2,3,4)
        ax6.plot(residuals_before,color='r',linestyle=':')
        ax6.plot(residuals_after, color='b',linestyle=':')
        ax6.plot(central_residuals_before,color='r')
        ax6.plot(central_residuals_after,color='b')
        ax6.plot(self.alignement_error[1],color='g')
        ax6.axhline(np.average(residuals_before),color='r',linestyle=':')
        ax6.axhline(np.average(residuals_after),color='b',linestyle=':')
        ax6.axhline(np.average(central_residuals_before),color='r')
        ax6.axhline(np.average(central_residuals_after),color='b')
        ax6.axhline(np.average(self.alignement_error[1]),color='g')

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(self.seq[i], cmap='Greys_r')
        ax2.set_title('Frame %i'%i)

        ax4 = fig.add_subplot(2, 3, 5)
        im = ax4.imshow(np.square(self.seq[i]-average),vmin=0, vmax=20)
        ax4.set_title('Residuals')
        plt.colorbar(im,orientation="horizontal")


        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(self.seq_trans[i],cmap='Greys_r')
        ax3.set_title('Transform')

        ax5 = fig.add_subplot(2, 3, 6)
        im = ax5.imshow(np.square(self.seq_trans[i]-average),vmin=0, vmax=20)
        plt.colorbar(im,orientation="horizontal")
        ax5.set_title('Residuals')
        fig.set_tight_layout(True)
        plt.show()

if __name__ == '__main__':

    filename = 'moon-00002.mp4'
    #filename = 'oaCapture-20150306-202356.avi'
    enhancer = CIA_enhance(filename,50,patch_size=8)
    enhancer.align_and_crop()
    enhancer.transform()
