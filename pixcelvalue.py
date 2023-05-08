import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import collections
from PIL import ImageTk, Image
import math

####################################################################################################################################################################################################################################################################################################################################################################################################################
##############################################################--------------THE CROPPED IMAGE----------------############################################################################################################################################################################################################################################################################################

def crop(im, base, angle, height, width):
    """Return a new, cropped image.

    Args:
        im: a PIL.Image instance
        base: a (x,y) tuple for the upper left point of the cropped area
        angle: angle, in radians, for which the cropped area should be rotated
        height: height in pixels of cropped area
        width: width in pixels of cropped area
    """
    base = Point(*base)
    points = getRotatedRectanglePoints(angle, base, height, width)
    return _cropWithPoints(im, angle, points)


def _cropWithPoints(im, angle, points):
    bounds = getBounds(points)
    im2 = im.crop(roundint(bounds))
    bound_center = getBoundsCenter(bounds)
    crop_center = getCenter(im2)
    # in the cropped image, this is where our points are
    crop_points = [pt.recenter(bound_center, crop_center) for pt in points]
    # this is where the rotated points would end up without expansion
    rotated_points = [pt.rotate(crop_center, angle) for pt in crop_points]
    # expand is necessary so that we don't lose any part of the picture
    im3 = im2.rotate(-angle * 180 / math.pi, expand=True)
    # but, since the image has been expanded, we need to recenter
    im3_center = getCenter(im3)
    rotated_expanded_points = [pt.recenter(crop_center, im3_center) for pt in rotated_points]
    im4 = im3.crop(roundint(getBounds(rotated_expanded_points)))
    return im4


def getCenter(im):
    return Point(*(d / 2 for d in im.size))


Bound = collections.namedtuple('Bound', ('left', 'upper', 'right', 'lower'))


def getBounds(points):
    xs, ys = zip(*points)
    # left, upper, right, lower using the usual image coordinate system
    # where top-left of the image is 0, 0
    return Bound(min(xs), min(ys), max(xs), max(ys))


def getBoundsCenter(bounds):
    return Point(
        (bounds.right - bounds.left) / 2 + bounds.left,
        (bounds.lower - bounds.upper) / 2 + bounds.upper
    )


def roundint(values):
    return tuple(int(round(v)) for v in values)


def getRotatedRectanglePoints(angle, base_point, height, width):
    # base_point is the upper left (ul)
    ur = Point(
        width * math.cos(angle),
        -width * math.sin(angle)
    )
    lr = Point(
        ur.x + height * math.sin(angle),
        ur.y + height * math.cos(angle)
    )
    ll = Point(
        height * math.cos(math.pi / 2 - angle),
        height * math.sin(math.pi / 2 - angle)
    )
    return tuple(base_point + pt for pt in (Point(0, 0), ur, lr, ll))



#pylint: disable=invalid-name


_Point = collections.namedtuple('Point', ['x', 'y'])


class Point(_Point):
    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def recenter(self, old_center, new_center):
        return self + (new_center - old_center)

    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/rotate.htm
    def rotate(self, center, angle):
        # angle should be in radians
        x = math.cos(angle) * (self.x - center.x) - math.sin(angle) * (self.y - center.y) + center.x
        y = math.sin(angle) * (self.x - center.x) + math.cos(angle) * (self.y - center.y) + center.y
        return Point(x, y)




################################################################################################################################################################################

def show_image_thresholde(image_acropped):
    
    im = Image.open(image_acropped)
    
    angle = math.pi / 3.5
    #base = (1, 700)
    base = (90, 820)
    height = 250
    #width = 850
    width = 1050
    cropped_im1 = crop(im, base, angle, height, width)
   
    cropped_im= np.array(cropped_im1)
    print(cropped_im.shape[0])

    cropped_im1 = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2GRAY)
    a=2
##    kernel1 = np.array([[a,  .2,  -a],
##                        [a,  .3,  -a],
##	                [a,  .2,  -a]
##                                    ])
##    ##	 
##    #np.transpose
##    _im = cv2.filter2D(src=cropped_im, ddepth=-1, kernel=np.transpose(kernel1))
##    #_im = cv2.filter2D(src=_im, ddepth=-1, kernel=np.transpose(kernel1))
##    #cv2.imshow("FILTRED IMAGE",_im)
##
##    #cropped_im = cv2.bitwise_not(cropped_im)
##
##    cropped_im = cv2.adaptiveThreshold(cropped_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 21, -2)
##    cropped_im = np.copy(cropped_im) 
##
##    # Specify size on horizontal axis
##    cols = cropped_im.shape[1]
##    horizontal_size = cols //50
##
##    # Create structure element for extracting horizontal lines through morphology operations
##    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
##    
##    # Apply morphology operations
##    cropped_im = cv2.erode(cropped_im, horizontalStructure)
##    
##    cropped_im = cv2.dilate(cropped_im, horizontalStructure)
##    
##    cv2.imshow("IMAGE2",cropped_im)
##    
##    #print(cropped_im)
##    cv2.waitKey(1) # waits until a key is pressed
    return cropped_im1
    
###################################################################################################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################################################################################################




def draw_from_video(video_path):
        vid = cv2.VideoCapture(video_path)
        print("video read")
        while True:
                _, img = vid.read()
                #print(img)

                try:
##                    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
##                    img = show_image_thresholde(original_image)
                    #cv2.imshow("IMAGE",np.array(img))
                    
                    img= np.array(img)
                    #img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #print(img.shape[1])
                    i=0
                    ilist=[]
                    pixcellist=[]
                    LEFT8ARRAY=[]
                    RIGHT8ARRAY=[]

                    for cunt,pixcel in enumerate(img[:,500]):
                                i+=1
                                ilist.append(i)
                                #print (pixcel)

                                if 60<pixcel<130 and(cunt<400):
                                        print("left")
                                if 60<pixcel<130 and(cunt>600):
                                        print("right")
                                if 60<pixcel<130 and 400<cunt and cunt<600 :
                                        print("good")
                                pixcellist.append(-pixcel)
##                                if np.any(img[100:200,500]>100):
##                                        print("right")
##                                else:
##                                        print("left")
                
                    plt.plot(ilist,pixcellist)
                    #plt.plot(ilist,RIGHT8ARRAY)
                    #plt.plot(ilist,LEFT8ARRAY)
                    plt.pause(0.0001)
                    plt.clf()
                except Exception as e:
                        print(e)
                        break
                        
def draw_from_image(path):
        for img in glob.glob(path):
##                img = show_image_thresholde(image)
                
                
                img= cv2.imread(img)
                img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #cv2.imshow("IMAGE",np.array(img))
                #print(img.shape[1])
                i=0
                ilist=[]
                pixcellist=[]
                
                for cunt,pixcel in enumerate(img[:,500]):
                        #print(pixcel)
                        i+=1
                        ilist.append(i)
                        pixcellist.append(-pixcel)
                        if 100<pixcel<140 and(cunt<100):
                                        print("left")
                        if 100<pixcel<140 and(cunt>300):
                                        print("right")
                        if 100<pixcel<140 and 100<cunt and cunt<300 :
                                        print("good")
                        
                        
                
               
                plt.plot(ilist,pixcellist)
                plt.pause(0.0001)
                plt.clf()
                
image_path = r'C:\Users\YASSINE\Desktop\XML\IMAGE A ANNOTTER\dataset dl a annoter\VIDEO RECORDED  04-16-33-04-10-20_clip/*.jpg'
video_path=r'C:\Users\111\Desktop\REPORT_APPS\SCRIPTS\AAC_DL.avi'
#draw_from_image(image_path)
draw_from_video(video_path)
