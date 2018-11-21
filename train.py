import tensorflow as tf
from net.Siamese import SiameseRPN
from utils.image_reader import Image_reader
from module.loss_module import Loss_op
import os
import numpy as np
import cv2
from module.gen_ancor import Anchor
from module.debug import debug
from config import cfg
class Train():
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.reader=Image_reader(cfg.root_dir)
        self.step_num=self.reader.img_num*cfg.epoch_num
        self.save_per_epoch=self.reader.img_num
        print(self.reader.img_num)
        self.loss_op=Loss_op()
        self.learning_rate=cfg.learning_rate
        self.decay_rate=cfg.decay_rate
        self.decay_step=cfg.decay_step
        self.model_dir=cfg.model_dir
        self.pre_trained_dir=cfg.pre_trained_dir
        self.anchor_op=Anchor(17,17)
        self.is_debug=False

    def train(self):
        template=tf.placeholder(dtype=tf.float32,shape=[1,127,127,3])
        detection=tf.placeholder(dtype=tf.float32,shape=[1,255,255,3])
        gt_box=tf.placeholder(dtype=tf.float32,shape=[4])
        step_t=tf.placeholder(dtype=tf.int64)

        net=SiameseRPN({'template':template,'detection':detection})

        pre_cls=net.layers['cls']
        pre_reg=net.layers['reg']

        cls_loss,reg_loss,label,target_box=self.loss_op.loss(gt_box,pre_cls,pre_reg)
        loss=cls_loss+reg_loss

        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
        debug_pre_cls=tf.nn.softmax(pre_cls)
        debug_pre_reg=pre_reg
        debug_pre_score=tf.nn.softmax(tf.reshape(pre_cls,(-1,2)))
        debug_pre_box=tf.reshape(pre_reg,(-1,4))
        #+++++++++++++++++++++debug++++++++++++++++++++++++++++++

        saver=tf.train.Saver(max_to_keep=50)
        with tf.name_scope('train_op'):
            global_step=tf.Variable(0,trainable=False)
            lr=tf.train.exponential_decay(self.learning_rate,global_step,self.decay_step,self.decay_rate,staircase=True,name='lr')
            train_op=tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,global_step)


        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.InteractiveSession(config=config)
        sess.run(tf.global_variables_initializer())

        if self.load(saver,sess,self.model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        epoch=0
        for step in range(self.step_num):
            template_p,_,detection_p,detection_label_p,offset,ratio,_,_=self.reader.get_data()
            template_p=np.expand_dims(template_p,axis=0)
            detection_p=np.expand_dims(detection_p,axis=0)
            feed_dict={template:template_p,detection:detection_p,gt_box:detection_label_p,step_t:step}

            #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
            if self.is_debug:
                _,loss_,cls_loss_,reg_loss_,lr_,debug_pre_cls_,debug_pre_reg_,debug_pre_score_,debug_pre_box_,label_,target_box_=\
                sess.run([train_op,loss,cls_loss,reg_loss,lr,debug_pre_cls,debug_pre_reg,debug_pre_score,debug_pre_box,label,target_box],feed_dict=feed_dict)
                if step %1000==0:
                    debug(detection_p[0],detection_label_p,debug_pre_cls_,debug_pre_reg_,debug_pre_score_,debug_pre_box_,label_,target_box_,step,self.anchor_op)
            #+++++++++++++++++++++debug++++++++++++++++++++++++++++++
            else:
                _,loss_,cls_loss_,reg_loss_,lr_=sess.run([train_op,loss,cls_loss,reg_loss,lr],feed_dict=feed_dict)
            if step %100==0:
                print('step={},loss={},cls_loss={},reg_loss={},lr={}'.format(step,loss_,cls_loss_,reg_loss_,lr_))

            if step %self.save_per_epoch==0 and step>0:
                self.save(saver,sess,self.model_dir,epoch)
                epoch+=1
    def load(self,saver,sess,ckpt_path):
        ckpt=tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess,os.path.join(ckpt_path,ckpt_name))
            print("Restored model parameters from {}".format(ckpt_name))
            return True
        else:
            return False
    def save(self,saver,sess,ckpt_path,epoch):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        model_path=os.path.join(ckpt_path,'model')
        saver.save(sess,model_path,epoch)
        print('saved model')
if __name__=='__main__':
    t=Train()
    t.train()


