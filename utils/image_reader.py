import numpy as np
import cv2
import os
class Image_reader():
    """docstring for Image_reader"""
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.interval=30
        self.cate_list={}
        self.cate_box={}
        self.img_num=0
        with open(os.path.join(self.root_dir,'list.txt')) as f:
            line=f.readline().strip('\n')
            while (line):
                #===========label===============
                with open(os.path.join(self.root_dir,line,'groundtruth.txt')) as f2:
                    line2=f2.readline().strip('\n')
                    boxes=[]
                    while (line2):
                        box=line2.split(',')
                        boxes.append([int(float(box[0])),int(float(box[1])),int(float(box[2])),int(float(box[3]))])
                        line2=f2.readline().strip('\n')
                    self.cate_box[line]=np.array(boxes)
                #===========label===============

                #===========img_list============
                img_list=[x for x in os.listdir(os.path.join(self.root_dir,line)) if 'jpg' in x]
                img_list.sort()
                self.cate_list[line]=np.array(img_list)
                #===========img_list============

                #=============filter============
                index=[]
                for i in range(len(self.cate_list[line])):
                    if not np.all(self.cate_box[line][i]==[0,0,0,0]):
                        #print('exception '+line+' '+str(i))
                        index.append(i)
                self.cate_box[line]=self.cate_box[line][index]
                self.cate_list[line]=self.cate_list[line][index]
                self.img_num+=len(self.cate_list[line])
                #=============filter============
                line=f.readline().strip('\n')
        print(self.img_num)

    def get_data(self,batch_size=1):
        cate=np.random.choice(list(self.cate_list.keys()))
        img_list=self.cate_list[cate]
        label_list=self.cate_box[cate]

        index_t=np.random.choice(range(len(img_list)))
        interval=np.random.choice(range(30,100))
        index_d=[index_t-interval if index_t-interval>0 else index_t+interval,index_t+interval if index_t+interval<len(img_list) else index_t-interval]
        index_d=np.random.choice(index_d)

        template=cv2.imread(os.path.join(self.root_dir,cate,img_list[index_t]))
        detection=cv2.imread(os.path.join(self.root_dir,cate,img_list[index_d]))

        template_label=label_list[index_t]
        detection_label=label_list[index_d]

        template_p,template_label_p,_,_=self.crop_resize(template,template_label,1)
        detection_p,detection_label_p,offset,ratio=self.crop_resize(detection,detection_label,2)

        return template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label

    def crop_resize(self,img,label,rate=1):
        #label=[x,y,w,h]===x,y is left-top corner
        x,y,w,h=label
        heigh,width=img.shape[:2]
        img=img.astype(np.float32)
        #mean_axis=np.mean(np.mean(img,0),0).astype(np.float32)
        mean_axis=np.mean(img,axis=(0,1)).astype(np.float32)
        p=(w+h)/2
        s=(w+p)*(h+p)
        side=round(np.sqrt(s)*rate)

        x1=int(x-int((side-w)/2))
        y1=int(y-int((side-h)/2))
        x2=int(x1+side)
        y2=int(y1+side)

        offset=np.zeros(2)
        offset[0]=x1-0
        offset[1]=y1-0

        if x1<0:
            img_offset=np.zeros((img.shape[0],-x1,3))+mean_axis
            img=np.hstack([img_offset,img])
            x-=x1
        if y1<0:
            img_offset=np.zeros((-y1,img.shape[1],3))+mean_axis
            img=np.vstack([img_offset,img])
            y-=y1
        if x2>width:
            img_offset=np.zeros((img.shape[0],x2-width+1,3))+mean_axis
            img=np.hstack([img,img_offset])
        if y2>heigh:
            img_offset=np.zeros((y2-heigh+1,img.shape[1],3))+mean_axis
            img=np.vstack([img,img_offset])

        x1=int(x-int((side-w)/2))
        y1=int(y-int((side-h)/2))
        x2=int(x1+side)
        y2=int(y1+side)
        crop_img=img[y1:y2,x1:x2,:]


        assert crop_img.shape[0]==side
        assert crop_img.shape[1]==side
        if rate==1:
            resize_img=cv2.resize(crop_img,(127,127))/255.
            ratio=side/127
            label=np.array([63,63,w/ratio,h/ratio]).astype(np.int32)
            label=label.astype(np.float32)
        if rate==2:
            resize_img=cv2.resize(crop_img,(255,255))/255.
            ratio=side/255
            label=np.array([127,127,w/ratio,h/ratio]).astype(np.int32)
            label=label.astype(np.float32)

        return resize_img,label,offset,ratio

if __name__=='__main__':
    reader=Image_reader('../data/VID')
    for i in range(20):
        template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label=reader.get_data()
        box=np.zeros(4)

        detection_label_p[2]=detection_label_p[2]*ratio
        detection_label_p[3]=detection_label_p[3]*ratio
        detection_label_p[0]=detection_label_p[0]*ratio+offset[0]
        detection_label_p[1]=detection_label_p[1]*ratio+offset[1]

        box[0]=int(detection_label_p[0]-(detection_label_p[2]-1)/2)
        box[1]=int(detection_label_p[1]-(detection_label_p[3]-1)/2)
        box[2]=round(detection_label_p[0]+(detection_label_p[2])/2)
        box[3]=round(detection_label_p[1]+(detection_label_p[3])/2)

        img=(detection).astype(np.uint8)

        detection_label[2]=detection_label[0]+detection_label[2]
        detection_label[3]=detection_label[1]+detection_label[3]

        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
        cv2.rectangle(img,(int(detection_label[0]),int(detection_label[1])),(int(detection_label[2]),int(detection_label[3])),(0,0,255),1)
        cv2.imshow('img',img)
        cv2.imshow('img2',(template_p*255).astype(np.uint8))
        cv2.waitKey(0)




