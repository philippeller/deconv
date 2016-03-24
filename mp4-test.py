import imageio
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import zoom
from scipy.ndimage import geometric_transform
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import map_coordinates


class CIA_enhance(object):
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
        self.patch_size = 16
        self.crop = self.patch_size * 12

    def align_and_crop(self):
        self.seq = []
        self.alignement_error = []
        for i in xrange(0,self.n_frames):
            # greyscale conversion
            img = np.average(self.reader.get_data(i),axis=2)
            for n in range(2):
                com = ndimage.measurements.center_of_mass(img)
                for axis in [0,1]:
                    img = np.roll(img, int(round(self.center[axis]-com[axis])), axis=axis)
            com = np.array(ndimage.measurements.center_of_mass(img))
            self.alignement_error.append(np.sum(np.square(com-self.center)))
            # crop
            lower_bounds = self.center - self.crop
            upper_bounds = self.center + self.crop
            img_c = img[lower_bounds[0]:upper_bounds[0],lower_bounds[1]:upper_bounds[1]]
            com = np.array(ndimage.measurements.center_of_mass(img_c))
            self.seq.append(img_c)
        
        print self.alignement_error


    def transform(self):

        grid_edges = np.arange(0,2*self.crop+self.patch_size,self.patch_size)
        grid_centers = 0.5*(grid_edges[1:]+grid_edges[:-1])
        xx, yy = np.meshgrid(grid_centers,grid_centers)
        #writer = imageio.get_writer('aligned_croped.mp4',fps=fps)

        self.seq = np.array(self.seq)
        average = np.average(self.seq,axis=0)

        grid_x, grid_y = np.mgrid[0:2*self.crop-1:np.complex(2*self.crop), 0:2*self.crop-1:np.complex(2*self.crop)]

        self.seq_trans = []

        for i in xrange(len(self.seq)):
            source = []
            destination = []

            xx_out = []
            yy_out = []

            for x in xrange(1,len(grid_centers)-1):
                for y in xrange(1,len(grid_centers)-1):
                    destination.append([grid_centers[x],grid_centers[y]])
                    x0,x1 = grid_edges[x:x+2]
                    y0,y1 = grid_edges[y:y+2]
                    patch = self.seq[i][x0:x1,y0:y1]
                    res = []
                    for move_x in xrange(-8,8):
                        for move_y in xrange(-8,8):
                            ref_x0 = x0+move_x
                            ref_x1 = x1+move_x
                            ref_y0 = y0+move_y
                            ref_y1 = y1+move_y
                            ref_patch = average[ref_x0:ref_x1,ref_y0:ref_y1]
                            res.append(np.sum(np.square(patch-ref_patch)))
                    minimum = np.argmin(res)
                    x_min = minimum/self.patch_size-8
                    y_min = minimum%self.patch_size-8
                    x_out = grid_centers[x]-x_min
                    y_out = grid_centers[y]-y_min
                    xx_out.append(x_out)
                    yy_out.append(y_out)
                    source.append([x_out, y_out])

            source = np.array(source)
            destination = np.array(destination)

            grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')

            trans = []
            for grid_z_p in grid_z:
                trans.append(map_coordinates(self.seq[i], grid_z_p.T,mode='reflect'))
            self.seq_trans.append(np.array(trans))

        self.seq_trans = np.array(self.seq_trans)
        average_trans = np.average(self.seq_trans,axis=0)

        fig = plt.figure(figsize=(15,15), frameon=False)
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(average, cmap='Greys_r')
        ax1.set_xlim(grid_edges[0],grid_edges[-1])
        ax1.set_ylim(grid_edges[0],grid_edges[-1])
        ax1.set_title('Average')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(average_trans, cmap='Greys_r')
        ax2.set_xlim(grid_edges[0],grid_edges[-1])
        ax2.set_ylim(grid_edges[0],grid_edges[-1])
        ax2.set_title('Average Transformed')
        writer = imageio.get_writer('average.jpeg',quality=100)
        writer.append_data(average)
        writer.close()
        writer = imageio.get_writer('average.tif')
        writer.append_data(average)
        writer.close()
        writer = imageio.get_writer('taverage.jpeg',quality=100)
        writer.append_data(average_trans)
        writer.close()
        writer = imageio.get_writer('taverage.tif')
        writer.append_data(average_trans)
        writer.close()

#fig = plt.figure(figsize=(15,15), frameon=False)
#fig.subplots_adjust(hspace=0)
#fig.subplots_adjust(wspace=0)
#ax1 = fig.add_subplot(2, 3, 1)
#ax1.imshow(average, cmap='Greys_r')
##ax1.scatter(xx.flat,yy.flat,marker='+',color='r',alpha=1.0)
#ax1.set_xlim(grid_edges[0],grid_edges[-1])
#ax1.set_ylim(grid_edges[0],grid_edges[-1])
#ax1.set_title('Average')
##ax1.contour(average)
#
#ax2 = fig.add_subplot(2, 3, 2)
#ax2.set_xlim(grid_edges[0],grid_edges[-1])
#ax2.set_ylim(grid_edges[0],grid_edges[-1])
#ax2.imshow(self.seq[i], cmap='Greys_r')
#ax2.set_title('Frame %i'%i)
##ax2.scatter(xx.flat,yy.flat,marker='+',color='r',alpha=1.0)
##ax2.scatter(xx_out,yy_out,marker='+',color='b',alpha=1.0)
#
#ax4 = fig.add_subplot(2, 3, 5)
#ax4.set_xlim(grid_edges[0],grid_edges[-1])
#ax4.set_ylim(grid_edges[0],grid_edges[-1])
#im = ax4.imshow(np.square(self.seq[i]-average),vmin=0, vmax=20)
##ax4.scatter(xx.flat,yy.flat,marker='+',color='r',alpha=1.0)
#plt.colorbar(im,orientation="horizontal")
#
#
#ax3 = fig.add_subplot(2, 3, 3)
#ax3.set_xlim(grid_edges[0],grid_edges[-1])
#ax3.set_ylim(grid_edges[0],grid_edges[-1])
#ax3.imshow(trans,cmap='Greys_r')
#ax3.set_title('Transform')
##ax3.scatter(xx.flat,yy.flat,marker='+',color='r',alpha=1.0)
#
#ax5 = fig.add_subplot(2, 3, 6)
#ax5.set_xlim(grid_edges[0],grid_edges[-1])
#ax5.set_ylim(grid_edges[0],grid_edges[-1])
#im = ax5.imshow(np.square(trans-average),vmin=0, vmax=20)
##ax5.scatter(xx.flat,yy.flat,marker='+',color='r',alpha=1.0)
#plt.colorbar(im,orientation="horizontal")
##ax2.contour(average)
        plt.show()

if __name__ == '__main__':

    filename = 'moon-00002.mp4'
    #filename = 'oaCapture-20150306-202356.avi'
    enhancer = CIA_enhance(filename,5)
    enhancer.align_and_crop()
    enhancer.transform()
