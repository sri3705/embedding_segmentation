from PIL import Image
import math
from operator import add

def delete_module(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    passj

class MyImage:
    def __init__(self, path):
        self.img = Image.open(path)
        self.size = self.img.size

    def getcolors(self):
        #return self.img.convert('RGB').getcolors()
        colors = {}
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                colors[self.getpixel(x,y)] = True
        return colors.keys()

    def getpixel(self, i, j):
        return self.img.getpixel((i, j))

    def putpixel(self, i, j, color):
        self.img.putpixel((i,j), color)


    def save(self, path):
        self.img.save(path)



class Supervoxel(object):

    def __init__(self, ID):
        '''
        :param arg1: the id of the supervoxel (we use color-tuple -> (r,g,b))
        :type arg1: any hashable object (we use tuple)
        '''
        try:
            hash(ID)
        except TypeError:
            raise Exception('ID must be immutable (hashable)')
        self.ID = ID
        #TODO removed this part for memory efficiency
        #self.pixels = {} # frame -> set of (x,y)
        #self.colors_dict = {} # (x,y,f) -> (R, G, B) actual color in the frame
        #TODO
        self.overlap_count = 0 #number of overlapping pixels with ground thruth
        self.__initializeCenter()

    def __initializeCenter(self):
        self.sum_x = 0
        self.sum_y = 0
        self.sum_t = 0
        self.number_of_pixels = 0

    def merge(self, supervoxel):
        self.sum_x += supervoxel.sum_x
        self.sum_y += supervoxel.sum_y
        self.sum_t += supervoxel.sum_t
        self.number_of_pixels += supervoxel.number_of_pixels

    def addVoxel(self, x,y,t, color, label=0):
        #TODO Removed this part for memory efficiency
        #if t not in self.pixels.keys():
        #    self.pixels[t] = set()
        #self.pixels[t].add((x,y))
        #self.colors_dict[ (x, y, t) ] = color
        #TODO
        self.sum_x += x
        self.sum_y += y
        self.sum_t += t
        self.number_of_pixels += 1
        self.overlap_count += label

    def hasPixel(self, x,y,f):
        return (x,y) in self.pixels[f]

    def getOverlap(self):
        return self.overlap_count*1.0 / self.number_of_pixels

    def getPixelsAtFrame(self, f):
        return self.pixels[f]

    def availableFrames(self):
        return self.pixels.keys()

    def center(self):
        n = self.number_of_pixels
        return (self.sum_x/n, self.sum_y/n, self.sum_t/n)

    def __str__(self):
        return "Supervoxel [ID:"+str(self.ID)+ ", Center:"+str(self.center()) + "]"

    def __eq__(self, other):
        return self.ID == other.ID

    def __hash__(self):
        return hash(self.ID)

#    def __getstate__(self):
#        state = {attr:getattr(self,attr) for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self,attr))}
#        return state

#    def __setstate__(self, dic):
#        for key in dic:
#            setattr(self, key, dic[key])



class HistogramSupervoxel(Supervoxel):

    def __init__(self, ID):
        super(HistogramSupervoxel, self).__init__(ID)
        # self.__initializeHistogram()


    def __initializeHistogram(self):
        self.R_hist = [0 for i in xrange(256)]
        self.G_hist = [0 for i in xrange(256)]
        self.B_hist = [0 for i in xrange(256)]

    def __initializeFlow(self):
        self.ch1_hist = [0 for i in xrange(256)]
        self.ch2_hist = [0 for i in xrange(256)]
        # self.ch3_hist = [0 for i in xrange(256)]

    def __initializeFCN(self):
        self.fcn = [0 for i in xrange(21)]

    def merge(self, supervoxel):
        super(HistogramSupervoxel, self).merge(supervoxel)
        self.R_hist = map(add, self.R_hist, supervoxel.R_hist) 
        self.G_hist = map(add, self.G_hist, supervoxel.G_hist) 
        self.B_hist = map(add, self.B_hist, supervoxel.B_hist) 
        self.ch1_hist = map(add, self.ch1_hist, supervoxel.ch1_hist) 
        self.ch2_hist = map(add, self.ch2_hist, supervoxel.ch2_hist) 
        self.fcn = map(add, self.fcn, supervoxel.fcn) 

    def addVoxel(self, x,y,t, color, label=0):
        super(HistogramSupervoxel, self).addVoxel(x,y,t,color,label)
        try:
            self.__updateHistogram(color)
        except:
            self.__initializeHistogram()
            self.__updateHitsogram(color)

    def addOpticalFlow(self, flow):
        try:
            self.ch1_hist[flow[0]] +=1
            self.ch2_hist[flow[2]] +=1
        except:
            self.__initializeFlow()
            self.ch1_hist[flow[0]] +=1
            self.ch2_hist[flow[2]] +=1
        # self.ch3_hist[flow[2]] +=1

    def addFCN(self, fcn):
        try:
            self.fcn = map(add, self.fcn, fcn)
            # for i in xrange(21):
                # self.fcn[i]+= fcn[i]
        except:
            self.__initializeFCN()
            self.fcn = map(add, self.fcn, fcn)
            # for i in xrange(21):
                # self.fcn[i]+= fcn[i]

    def __updateHistogram(self, color):
        self.R_hist[color[0]] += 1
        self.G_hist[color[1]] += 1
        self.B_hist[color[2]] += 1

    def getFeature(self, number_of_bins=256):
        bin_width = 256/number_of_bins
        bin_num = -1
        r_hist = [0 for i in xrange(number_of_bins)]
        g_hist = r_hist[:]
        b_hist = r_hist[:]
        for i in xrange(256):
            if i%bin_width == 0:
                bin_num+=1
            r_hist[bin_num]+=self.R_hist[i]
            g_hist[bin_num]+=self.G_hist[i]
            b_hist[bin_num]+=self.B_hist[i]
        color_histogram = [i*1.0/self.number_of_pixels for i in r_hist+g_hist+b_hist]
        return color_histogram

    def getFCN(self):
        return [i*1.0/self.number_of_pixels for i in self.fcn]

    def getOpticalFlow(self,optical_flow_bins=256):
        bin_width = 256/optical_flow_bins
        bin_num = -1
        ch1_hist = [0 for i in xrange(optical_flow_bins)]
        ch2_hist = ch1_hist[:]
        for i in xrange(256):
            if i%bin_width == 0:
                    bin_num+=1
            ch1_hist[bin_num]+=self.ch1_hist[i]
            ch2_hist[bin_num]+=self.ch2_hist[i]
        optical_flow = [i*1.0/self.number_of_pixels for i in ch1_hist+ch2_hist]
        return optical_flow



