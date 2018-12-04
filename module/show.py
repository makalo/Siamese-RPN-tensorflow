import os
import numpy as np
import cv2
import time
class Visual():
    def __init__(self):
        self.img_path='../data/vot2013/car'
        self.label_path='../data/vot2013/car/groundtruth.txt'
    def show(self):
        img_list=[x for x in os.listdir(self.img_path) if 'jpg' in x or 'JPEG' in x]
        img_list.sort()
        f=open(self.label_path,'r')
        t=time.time()
        im=cv2.imread(os.path.join(self.img_path,img_list[0]))
        fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
        img_h,img_w,_=im.shape
        videoWriter=cv2.VideoWriter(os.path.join('../data/vedio','car2.mp4'),fourcc,30,(img_w,img_h))
        for img in img_list:
            line=f.readline().strip('\n')
            box=line.split(',')
            box=[int(float(box[0])),int(float(box[1])),int(float(box[2])),int(float(box[3]))]
            box[2]=box[0]+box[2]
            box[3]=box[1]+box[3]
            im=cv2.imread(os.path.join(self.img_path,img))
            videoWriter.write(im)
            cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)

            cv2.imshow('img',im)
            cv2.waitKey(10)
        f.close()
        cv2.destroyAllWindows()
        videoWriter.release()
        print(time.time()-t)
if __name__=='__main__':
    t=Visual()
    t.show()