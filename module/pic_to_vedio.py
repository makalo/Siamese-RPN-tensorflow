import sys
sys.path.append('./')
import os
import numpy as np
import cv2
from config import cfg
from utils.image_reader_forward import Image_reader
class Pic_vedio():
    def __init__(self):
        self.reader=Image_reader(img_path=cfg.img_path,label_path=cfg.label_path)
        self.vedio_dir=cfg.vedio_dir
        self.vedio_name=cfg.vedio_name
    def test(self):
        pre_box=[50.,50.,50.,50.]
        for step in range(self.reader.img_num):
            img,box_ori,img_p,box_p,offset,ratio=self.reader.get_data(frame_n=step,pre_box=pre_box)
            if step==0:
                fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
                img_h,img_w,_=img.shape
                videoWriter=cv2.VideoWriter(os.path.join(self.vedio_dir,self.vedio_name),fourcc,30,(img_w,img_h))
            else:
                videoWriter.write(img)
                cv2.imshow('img',img)
                cv2.waitKey(10)
        videoWriter.release()
        cv2.destroyAllWindows()
        print('vedio is saved in '+self.vedio_dir)
if __name__=='__main__':
    t=Pic_vedio()
    t.test()