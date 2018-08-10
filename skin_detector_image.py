import numpy as np
import argparse
import cv2
import colorsys
import matplotlib.pyplot as plt

def skin_detect(im):

    lower = np.array( [ 0,40,80], dtype= "uint8")
    upper = np.array( [ 20,255,255 ], dtype= "uint8")

    converted =cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    skinMask=cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    skinMask=cv2.erode(skinMask,kernel,iterations=2)
    skinMask=cv2.dilate(skinMask,kernel,iterations=2)

    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
    skin_bgr=cv2.bitwise_and(im,im,mask=skinMask)

#in HSV 255
    return(skin_bgr)

def screen(im):
    im_filtered=2*im-im*im
    return im_filtered

def bright_eval(skin):
    cnt=0
    sum=[0.0,0.0,0.0]
    print("skin shape: ",skin.shape)
    for i in range(skin.shape[0]):
        for j in range(skin.shape[1]):
            if not (skin[i,j]==[0,0,0]).all():
                if i==j:
                    print("sample pixel: ",skin[i,j])
                sum=sum+skin[i,j]
                cnt+=1
    print("sum and cnt: ",sum,cnt)
    avg=sum/cnt
    return (avg)

def sample_pixel(skin):
    pixels=[]
    for k in range(10):
        for i in range(skin.shape[0]):
            for j in range(skin.shape[1]):
                if skin[i,j].all!=0:
                    pixels.append(skin[i,j])
    return pixels

def add_screen(im,skin):
    ret , mask=cv2.threshold(skin,10,255,cv2.THRESH_BINARY)
    mask_inv=cv2.bitwise_not(mask)
    print(type(skin))
    skin_patch=cv2.bitwise_and(skin,skin,mask=mask)
    image=cv2.add(im,skin_patch)
    return(image)



def to_hsv( color ):
    """ converts color tuples to floats and then to hsv """
    color=colorsys.rgb_to_hsv(*[x/255.0 for x in color])
    print("color in hsv: ",color)
    return (color) #rgb_to_hsv wants floats!

def color_dist( c1, c2):
    """ returns the squared euklidian distance between two color vectors in hsv space """
    return sum( (a-b)**2 for a,b in zip(to_hsv(c1),to_hsv(c2)) )

def min_color_diff( avg, colors):
    """ returns the `(distance, color_name)` with the minimal distance to `colors`"""
    return min( # overal best is the best match to any color:
        (color_dist(avg, test), colors[test]) # (distance to `test` color, color name)
        for test in colors)


if __name__=="__main__":

    im=cv2.imread('9.jpg') # in BGR 255
#    colors =((196, 2, 51),(255, 165, 0),(255, 205, 0),(0, 128, 0),(0, 0, 255),(127, 0, 255),(0, 0, 0),(255, 255, 255))
#    colors=color_cvt(colors)
#    print(colors)

#in 255 RGB
    colors = dict((
    ((233,210,214), "1"),
    ((231,210,194), "2"),
    ((240,241,191), "3"),
    ((221,182,157), "4"),
    ((225,189,160), "5"),
    ((182,137,108), "6"),
    ((167,121,96), "7"),
    ((150,96,58), "8"),
    ((136,79,54),"9"),
    ((61,44,40),"10")))


    cv2.imshow('original',im)
    cv2.waitKey(0)
    skin_bgr=skin_detect(im) # 255 HSV
    cv2.imshow('maksed',skin_bgr)
    cv2.waitKey(0)

#    im=cv2.normalize(im,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) #0-1BGR
#    skin=cv2.normalize(skin,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)#01 HSV


#    skin_screen=screen(skin) #?
##    brightness=bright_eval(skin)
##    print(brightness)
#    cv2.imshow("screen",skin_screen)
#    cv2.waitKey(0)

    avg_bgr=bright_eval(skin_bgr) #01 HSV
#    color_show=255*np.asarray(colorsys.hsv_to_rgb(avg[0],avg[1],avg[2]))
#    print("avg in RGB 255: ",color_show)
#    avg=np.asarray(avg)


    print("avge color in bgr: ",avg_bgr)
    avg_rgb=np.flip(avg_bgr,0)
    print(min_color_diff(avg_rgb,colors))
    #cvt to 255 in RGB

#    plt.imshow([avg_rgb])
#    plt.show()
#    print(skin.shape[0])
#    im_add=add_screen(im,skin)

#    cv2.imshow("overlay",im_add)
#    cv2.waitKey(0)
    cv2.destroyAllWindows()


    





