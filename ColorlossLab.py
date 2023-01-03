import torch
import kornia
import cv2
#import cv2.cv2 as cv2

def ColorDifference(imgo,imgr):
    imglabo=cv2.cvtColor(imgo,cv2.COLOR_BGR2LAB)
    imglabr=cv2.cvtColor(imgr,cv2.COLOR_BGR2LAB)
    diff=(imglabr-imglabo)*(imglabr-imglabo)
    RD=diff[:,:,1]
    BD=diff[:,:,2]
    Result=RD+BD
    Result=cv2.blur(Result,(11,11))*0.001
    return Result

if __name__=="__main__":
    imgo=cv2.imread('orig1.png')
    imglabo=cv2.cvtColor(imgo,cv2.COLOR_BGR2LAB)

    imgr=cv2.imread('rec1.png')
    imglabr=cv2.cvtColor(imgr,cv2.COLOR_BGR2LAB)

    diff=(imglabr-imglabo)*(imglabr-imglabo)
    RD=diff[:,:,1]
    BD=diff[:,:,2]
    Result=RD+BD
    cv2.imwrite('Result.png',Result)
    pass


    pass




