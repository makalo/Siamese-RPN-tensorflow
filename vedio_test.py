import tensorflow as tf
from net.Siamese_forward import SiameseRPN
from utils.image_reader_forward import Image_reader
import os
import numpy as np
import cv2
from module.gen_ancor import Anchor
import time
import sys
from config import cfg
import imageio
class VedioTest():
    def __init__(self):
        self.reader=Image_reader(mode='vedio')
        self.model_dir=cfg.model_dir
        self.vedio_dir=cfg.vedio_dir
        self.vedio_name=cfg.vedio_name
        self.anchor_op=Anchor(17,17)
        self.anchors=self.anchor_op.anchors
        self.anchors=self.anchor_op.corner_to_center(self.anchors)
        self.penalty_k=cfg.penalty_k
        self.window_influence=cfg.window_influence
        self.lr=cfg.lr
        #===================init-parameter==================
        self.selectingObject = False
        self.initTracking = False
        self.onTracking = False
        self.ix, self.iy, self.cx, self.cy = -1, -1, -1, -1
        self.w, self.h = 0, 0
        self.inteval = 1
        self.duration = 0.01
        self.select=True
        #===================init-parameter==================
    def test(self):
        #===================input-output====================
        img_t=tf.placeholder(dtype=tf.float32,shape=[1,None,None,3])
        conv_c=tf.placeholder(dtype=tf.float32,shape=[4,4,256,10])
        conv_r=tf.placeholder(dtype=tf.float32,shape=[4,4,256,20])

        net=SiameseRPN({'img':img_t,'conv_c':conv_c,'conv_r':conv_r})

        pre_conv_c=net.layers['t_c_k']
        pre_conv_r=net.layers['t_r_k']

        pre_cls=net.layers['cls']
        pre_reg=net.layers['reg']
        pre_cls=tf.nn.softmax(tf.reshape(pre_cls,(-1,2)))
        pre_reg=tf.reshape(pre_reg,(-1,4))
        conv_r_=np.zeros((4,4,256,20))
        conv_c_=np.zeros((4,4,256,10))
        pre_box=None
        #===================input-output====================
<<<<<<< HEAD
=======

>>>>>>> e690ed5433117e707ff59f34ddd6f793a9c8807b
        #======================hanning======================
        window = np.outer(np.hanning(17), np.hanning(17))
        window=np.stack([window,window,window,window,window],-1)
        self.window=window.reshape((-1))
        #======================hanning======================
<<<<<<< HEAD
        
=======

>>>>>>> e690ed5433117e707ff59f34ddd6f793a9c8807b
        #================start-tensorflow===================
        loader=tf.train.Saver()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.InteractiveSession(config=config)
        sess.run(tf.global_variables_initializer())
        if self.load(loader,sess,self.model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        #================start-tensorflow===================

        #===================init-vedio======================
        if (len(sys.argv)==2):
            self.vedio_name=sys.argv[1]
        cap = cv2.VideoCapture(os.path.join(self.vedio_dir,self.vedio_name))
        cv2.namedWindow('tracking')
        cv2.setMouseCallback('tracking',self.draw_boundingbox)
        # fps =cap.get(cv2.CAP_PROP_FPS)
        # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
        # videoWriter=cv2.VideoWriter(os.path.join(self.vedio_dir,self.vedio_name.split('.')[0]+'_box.'+self.vedio_name.split('.')[1]),fourcc,fps,size)
        #===================init-vedio======================

        ret, frame = cap.read()
        frames=[]
        self.note=[]

        while(cap.isOpened()):
            if self.select:
                frame_temp=frame.copy()
                cv2.putText(frame_temp, 'select an area for tracing', (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            if(self.selectingObject):
                cv2.rectangle(frame,(self.ix,self.iy), (self.cx,self.cy), (0,255,255), 1)
            elif(self.initTracking):
                cv2.rectangle(frame,(self.ix,self.iy), (self.ix+self.w,self.iy+self.h), (0,255,255), 2)
                #videoWriter.write(frame)
                frames.append(frame[:,:,::-1])
                #===================init-net======================
                frame,box_ori,img_p,box_p,offset,ratio=self.reader.get_vedio_data(img=frame,box_ori=[self.ix,self.iy,self.w,self.h],frame_n=0)
                img_p=np.expand_dims(img_p,axis=0)
                feed_dict={img_t:img_p,conv_c:conv_c_,conv_r:conv_r_}
                conv_c_,conv_r_=sess.run([pre_conv_c,pre_conv_r],feed_dict=feed_dict)
                pre_box=box_ori#[x,y,self.w,self.h]===x,y is left-top corner
                self.note.append(np.array([box_ori[0]+box_ori[2]/2,box_ori[1]+box_ori[3]/2,box_ori[2],box_ori[3],1.0]))
                #===================init-net======================

                self.initTracking = False
                self.onTracking = True
                self.select= False
            elif(self.onTracking):
                #===================update-net======================
                frame,box_ori,img_p,box_p,offset,ratio=self.reader.get_vedio_data(img=frame,frame_n=1,pre_box=pre_box,note=self.note)
                img_p=np.expand_dims(img_p,axis=0)
                feed_dict={img_t:img_p,conv_c:conv_c_,conv_r:conv_r_}
                t0 = time.time()
                pre_cls_,pre_reg_=sess.run([pre_cls,pre_reg],feed_dict=feed_dict)
                t1 = time.time()
                bbox,score=self.nms(img_p[0],pre_cls_,pre_reg_,box_p)
                pre_box=self.recover(frame,bbox,offset,ratio,pre_box,score)#[x1,y1,x2,y2]

                frame=cv2.rectangle(frame,(int(pre_box[0]),int(pre_box[1])),(int(pre_box[2]),int(pre_box[3])),(0,0,255),1)

                pre_box[2]=pre_box[2]-pre_box[0]
                pre_box[3]=pre_box[3]-pre_box[1]

                self.duration = 0.8*self.duration + 0.2*(t1-t0)
                cv2.putText(frame, 'FPS: '+str(1/self.duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                #videoWriter.write(frame)
                frames.append(frame[:,:,::-1])
                #===================update-net======================
            cv2.imshow('tracking', frame)
            if self.select:
                frame=frame_temp
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            c = cv2.waitKey(self.inteval) & 0xFF
            if c==27 or c==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print('GIF and video are being synthesized.place wait for one minute..............')
        #videoWriter.release()
        #imageio.mimsave(os.path.join(self.vedio_dir,self.vedio_name.split('.')[0]+'_box.gif'), frames, 'GIF', duration=0.01)
        print('vedio is saved in '+self.vedio_dir)
    # mouse callback function
    def draw_boundingbox(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.selectingObject = True
            self.onTracking = False
            self.ix, self.iy = x, y
            self.cx, self.cy = x, y
            print(self.ix,self.iy)

        elif event == cv2.EVENT_MOUSEMOVE:
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.selectingObject = False
            if(abs(x-self.ix)>10 and abs(y-self.iy)>10):
                self.w, self.h = abs(x - self.ix), abs(y - self.iy)
                self.ix, self.iy = min(x, self.ix), min(y, self.iy)
                self.initTracking = True
            else:
                self.onTracking = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.onTracking = False
            if(self.w>0):
                self.ix, self.iy = x-self.w/2, y-self.h/2
                self.initTracking = True

    def nms(self,img,scores,delta,gt_p):
        img=(img*255).astype(np.uint8)
        target_sz=gt_p[2:]
        score=scores[:,1]
        # #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        # b=self.anchor_op.center_to_corner(gt_p.reshape((1,4)))
        # cv2.rectangle(img,(int(b[0][0]),int(b[0][1])),(int(b[0][2]),int(b[0][3])),(0,255,0),1)
        # #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        bboxes=np.zeros_like(delta)
        bboxes[:,0]=delta[:,0]*self.anchors[:,2]+self.anchors[:,0]
        bboxes[:,1]=delta[:,1]*self.anchors[:,3]+self.anchors[:,1]
        bboxes[:,2]=np.exp(delta[:,2])*self.anchors[:,2]
        bboxes[:,3]=np.exp(delta[:,3])*self.anchors[:,3]#[x,y,w,h]
        def change(r):
            return np.maximum(r, 1./r)
        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)
        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        s_c = change(sz(bboxes[:,2], bboxes[:,3]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (bboxes[:,2] / bboxes[:,3]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1.) * self.penalty_k)
        pscore = penalty * score


        # window float
        pscore = pscore * (1 - self.window_influence) + self.window * self.window_influence
        # #==================debug=====================
        # pscore = score
        # #==================debug=====================
        best_pscore_id = np.argmax(pscore)
        best_pscore = np.max(pscore)
        print(best_pscore)

        self.lr = penalty[best_pscore_id] * score[best_pscore_id] * self.lr
        bbox=bboxes[best_pscore_id].reshape((1,4))#[x,y,w,h]
        # #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        # b=self.anchor_op.center_to_corner(bbox)
        # cv2.rectangle(img,(int(b[0][0]),int(b[0][1])),(int(b[0][2]),int(b[0][3])),(255,0,0),1)
        # cv2.imshow('resize',img)
        # cv2.waitKey(0)
        # #+++++++++++++++++++++debug++++++++++++++++++++++++++++++

        return bbox[0],best_pscore

    def recover(self,img,box,offset,ratio,pre_box,score):
        #label=[c_x,c_y,w,h]
        box[2]=box[2]*ratio
        box[3]=box[3]*ratio
        box[0]=box[0]*ratio+offset[0]
        box[1]=box[1]*ratio+offset[1]


        if score<0.9:
            box[2] = pre_box[2]
            box[3] = pre_box[3]
        else:
            box[2] = pre_box[2] * (1 - self.lr) + box[2] * self.lr
            box[3] = pre_box[3] * (1 - self.lr) + box[3] * self.lr

        note=np.zeros((5),dtype=np.float32)
        note[0:4]=box
        note[4]=score
        self.note.append(note)

        box[0]=int(box[0]-(box[2]-1)/2)
        box[1]=int(box[1]-(box[3]-1)/2)
        box[2]=round(box[0]+(box[2]))
        box[3]=round(box[1]+(box[3]))
        # #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        # cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
        # cv2.imshow('ori',img)
        # cv2.waitKey(0)
        # #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        return box#[x1,y1,x2,y2]
    def load(self,saver,sess,ckpt_path):
        ckpt=tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess,os.path.join(ckpt_path,ckpt_name))
            print("Restored model parameters from {}".format(ckpt_name))
            return True
        else:
            return False

if __name__=='__main__':
    t=VedioTest()
    t.test()
