import tensorflow as tf
from net.Siamese_forward import SiameseRPN
from utils.image_reader_forward import Image_reader
import os
import numpy as np
import cv2
from module.gen_ancor import Anchor
from config import cfg
import imageio
class Test():
    def __init__(self):
        self.reader=Image_reader(img_path=cfg.img_path,label_path=cfg.label_path)
        self.model_dir=cfg.model_dir
        self.anchor_op=Anchor(17,17)
        self.anchors=self.anchor_op.anchors
        self.anchors=self.anchor_op.corner_to_center(self.anchors)
        self.penalty_k=cfg.penalty_k
        self.window_influence=cfg.window_influence
        self.lr=cfg.lr
        self.vedio_dir=cfg.vedio_dir
        self.vedio_name=cfg.vedio_name
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

        #======================hanning======================
        w = np.outer(np.hanning(17), np.hanning(17))
        w=np.stack([w,w,w,w,w],-1)
        self.window=w.reshape((-1))
        #======================hanning======================

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

        frames=[]
        for step in range(self.reader.img_num):
            img,box_ori,img_p,box_p,offset,ratio=self.reader.get_data(frame_n=step,pre_box=pre_box)
            #print(img.shape)

            img_p=np.expand_dims(img_p,axis=0)
            feed_dict={img_t:img_p,conv_c:conv_c_,conv_r:conv_r_}
            if step==0:
                fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
                img_h,img_w,_=img.shape
                videoWriter=cv2.VideoWriter(os.path.join(self.vedio_dir,self.vedio_name),fourcc,30,(img_w,img_h))
                videoWriter_box=cv2.VideoWriter(os.path.join(self.vedio_dir,self.vedio_name.split('.')[0]+'_box.'+self.vedio_name.split('.')[1]),fourcc,30,(img_w,img_h))
                #init
                conv_c_,conv_r_=sess.run([pre_conv_c,pre_conv_r],feed_dict=feed_dict)
                pre_box=box_ori#[x,y,w,h]===x,y is left-top corner
            else:
                frames.append(img[:,:,::-1])
                videoWriter.write(img)
                pre_cls_,pre_reg_=sess.run([pre_cls,pre_reg],feed_dict=feed_dict)
                bbox=self.nms(img_p[0],pre_cls_,pre_reg_,box_p)
                pre_box=self.recover(img,bbox,offset,ratio,pre_box)#[x1,y1,x2,y2]

                img=cv2.rectangle(img,(int(pre_box[0]),int(pre_box[1])),(int(pre_box[2]),int(pre_box[3])),(0,0,255),1)

<<<<<<< HEAD
                # #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++
                # box_ori[2]=box_ori[0]+box_ori[2]
                # box_ori[3]=box_ori[1]+box_ori[3]
                # img=cv2.rectangle(img,(int(box_ori[0]),int(box_ori[1])),(int(box_ori[2]),int(box_ori[3])),(0,0,0),1)
                # #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++
=======
                #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++
                # box_ori[2]=box_ori[0]+box_ori[2]
                # box_ori[3]=box_ori[1]+box_ori[3]
                # img=cv2.rectangle(img,(int(box_ori[0]),int(box_ori[1])),(int(box_ori[2]),int(box_ori[3])),(0,0,0),1)
                #+++++++++++++++++++++gt_box++++++++++++++++++++++++++++++
>>>>>>> e690ed5433117e707ff59f34ddd6f793a9c8807b

                cv2.imshow('img',img)
                cv2.waitKey(10)
                videoWriter_box.write(img)

                pre_box[2]=pre_box[2]-pre_box[0]
                pre_box[3]=pre_box[3]-pre_box[1]
        print('GIF and video are being synthesized.place wait for one minute..............')
        imageio.mimsave(os.path.join(self.vedio_dir,self.vedio_name.split('.')[0]+'_box.gif'), frames, 'GIF', duration=0.01)
        videoWriter.release()
        videoWriter_box.release()
        print('vedio is saved in '+self.vedio_dir)

    def nms(self,img,scores,delta,gt_p):
        img=(img*255).astype(np.uint8)
        target_sz=gt_p[2:]
        score=scores[:,1]

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
        best_pscore_id = np.argmax(pscore)

        self.lr = penalty[best_pscore_id] * score[best_pscore_id] * self.lr
        bbox=bboxes[best_pscore_id].reshape((1,4))#[x,y,w,h]

        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        # b=self.anchor_op.center_to_corner(bbox)
        # cv2.rectangle(img,(int(b[0][0]),int(b[0][1])),(int(b[0][2]),int(b[0][3])),(255,0,0),1)
        # cv2.imshow('resize',img)
        # cv2.waitKey(0)
        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++

        return bbox[0]

    def recover(self,img,box,offset,ratio,pre_box):
        #label=[c_x,c_y,w,h]
        box[2]=box[2]*ratio
        box[3]=box[3]*ratio
        box[0]=box[0]*ratio+offset[0]
        box[1]=box[1]*ratio+offset[1]

        box[2] = pre_box[2] * (1 - self.lr) + box[2] * self.lr
        box[3] = pre_box[3] * (1 - self.lr) + box[3] * self.lr

        box[0]=int(box[0]-(box[2]-1)/2)
        box[1]=int(box[1]-(box[3]-1)/2)
        box[2]=round(box[0]+(box[2]))
        box[3]=round(box[1]+(box[3]))

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
    t=Test()
<<<<<<< HEAD
    t.test()
=======
    t.test()
>>>>>>> e690ed5433117e707ff59f34ddd6f793a9c8807b
