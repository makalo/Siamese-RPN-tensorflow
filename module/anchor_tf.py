import tensorflow as tf
class Anchor_tf():
    def __init__(self):
        self.width=255
        self.height=255
    def center_to_corner(self,box):
        t_1=box[:,0]-(box[:,2]-1)/2
        t_2=box[:,1]-(box[:,3]-1)/2
        t_3=box[:,0]+(box[:,2]-1)/2
        t_4=box[:,1]+(box[:,3]-1)/2
        box_temp=tf.transpose(tf.stack([t_1,t_2,t_3,t_4],axis=0),(1,0))
        return box_temp
    def corner_to_center(self,box):
        t_1=box[:,0]+(box[:,2]-box[:,0])/2
        t_2=box[:,1]+(box[:,3]-box[:,1])/2
        t_3=(box[:,2]-box[:,0])
        t_4=(box[:,3]-box[:,1])
        box_temp=tf.transpose(tf.stack([t_1,t_2,t_3,t_4],axis=0),(1,0))
        return box_temp
    def diff_anchor_gt(self,gt,anchors):
        #gt [x,y,w,h]
        #anchors [x,y,w,h]
        t_1=(gt[0]-anchors[:,0])/(anchors[:,2]+0.01)
        t_2=(gt[1]-anchors[:,1])/(anchors[:,3]+0.01)
        t_3=tf.log(gt[2]/(anchors[:,2]+0.01))
        t_4=tf.log(gt[3]/(anchors[:,3]+0.01))
        diff_anchors=tf.transpose(tf.stack([t_1,t_2,t_3,t_4],axis=0),(1,0))
        return diff_anchors#[dx,dy,dw,dh]
    def iou(self,box1,box2):
        """ Intersection over Union (iou)
            Args:
                box1 : [N,4]
                box2 : [K,4]
                box_type:[x1,y1,x2,y2]
            Returns:
                iou:[N,K]
        """
        N=box1.get_shape()[0]
        K=box2.get_shape()[0]
        box1=tf.reshape(box1,(N,1,4))+tf.zeros((1,K,4))#box1=[N,K,4]
        box2=tf.reshape(box2,(1,K,4))+tf.zeros((N,1,4))#box1=[N,K,4]
        x_max=tf.reduce_max(tf.stack((box1[:,:,0],box2[:,:,0]),axis=-1),axis=2)
        x_min=tf.reduce_min(tf.stack((box1[:,:,2],box2[:,:,2]),axis=-1),axis=2)
        y_max=tf.reduce_max(tf.stack((box1[:,:,1],box2[:,:,1]),axis=-1),axis=2)
        y_min=tf.reduce_min(tf.stack((box1[:,:,3],box2[:,:,3]),axis=-1),axis=2)
        tb=x_min-x_max
        lr=y_min-y_max
        zeros=tf.zeros_like(tb)
        tb=tf.where(tf.less(tb,0),zeros,tb)
        lr=tf.where(tf.less(lr,0),zeros,lr)
        over_square=tb*lr
        all_square=(box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])+(box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])-over_square
        return over_square/all_square
    def pos_neg_anchor2(self,gt,anchors):
        #gt [x,y,w,h]
        #anchors [x1,y1,x2,y2]
        zeros=tf.zeros_like(anchors)
        ones=tf.ones_like(anchors)
        all_box=tf.where(tf.less(anchors,0),zeros,anchors)
        all_box=tf.where(tf.greater(all_box,self.width-1),ones*(self.width-1),all_box)
        target_box=tf.zeros((anchors.get_shape()[0],4),dtype=tf.float32)
        target_inside_weight_box=tf.zeros((anchors.get_shape()[0],4),dtype=tf.float32)
        target_outside_weight_box=tf.ones((anchors.get_shape()[0],4),dtype=tf.float32)
        label=-tf.ones((anchors.get_shape()[0],),dtype=tf.float32)


        gt_array=tf.reshape(gt,(1,4))
        gt_array=self.center_to_corner(gt_array)

        iou_value=tf.reshape(self.iou(all_box,gt_array),[-1])

        pos_value,pos_index=tf.nn.top_k(iou_value,16)
        pos_mask_label=tf.ones_like(label)
        label=tf.where(tf.greater_equal(iou_value,pos_value[-1]),pos_mask_label,label)

        neg_index=tf.reshape(tf.where(tf.less(iou_value,0.3)),[-1])
        neg_index=tf.random_shuffle(neg_index)
        neg_index=neg_index[0:48]
        neg_index=tf.reduce_sum(tf.one_hot(neg_index,anchors.get_shape()[0]),axis=0)
        neg_mask_label=tf.zeros_like(label)
        label=tf.where(tf.equal(neg_index,1),neg_mask_label,label)

        target_box=self.diff_anchor_gt(gt,self.corner_to_center(all_box))
        temp_label=tf.transpose(tf.stack([label,label,label,label],axis=0),(1,0))
        target_inside_weight_box=tf.where(tf.equal(temp_label,1),temp_label,target_inside_weight_box)
        target_outside_weight_box=target_outside_weight_box*1.0/48.
        #print(target_outside_weight_box[np.where(target_outside_weight_box>0)])
        return label,target_box,target_inside_weight_box,target_outside_weight_box,all_box


if __name__=='__main__':
    import sys
    sys.path.append('../')
    from utils.image_reader_cuda import Image_reader
    from module.gen_ancor import Anchor
    import numpy as np
    import cv2
    anchors_op=Anchor(17,17)
    anchors=anchors_op.anchors
    reader=Image_reader('../data/VID_ALL')
    test=Anchor_tf()
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess=tf.InteractiveSession(config=config)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord,sess=sess)
    for i in range(30):
        #template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label=reader.get_data()
        template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label,index_t,index_d=\
        reader.template_p,reader.template_label_p,reader.detection_p,reader.detection_label_p,reader.offset,reader.ratio,reader.detection,reader.detection_label,reader.index_t,reader.index_d

        template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label,index_t,index_d=\
        sess.run([template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label,index_t,index_d])

        img=np.ones((255,255,3),dtype=np.uint8)*255
        img=(detection_p*255).astype(np.uint8)
    #===========debug_all===============================
        #gt=[100,100,50,50]
        gt=detection_label_p
        gt_array=np.array(gt).reshape((1,4))
        gt_array=anchors_op.center_to_corner(gt_array)[0]
        label,target_box,target_inside_weight_box,_,all_box=test.pos_neg_anchor2(tf.convert_to_tensor(gt),tf.convert_to_tensor(anchors))
        label,target_box,target_inside_weight_box,all_box=sess.run([label,target_box,target_inside_weight_box,all_box])
        #print(target_box[np.where(label==1)])
        #negtive
        index=np.where(label==0)
        boxes=all_box[index]
        for b in boxes:
            cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(255,0,0),1)
        #positive
        index=np.where(label==1)
        boxes=all_box[index]
        for b in boxes:
            cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,255,0),1)
        #target_box
        anchor=anchors_op.regu()
        anchor=anchors_op.corner_to_center(anchor)
        boxes=np.zeros_like(target_box)
        boxes[:,0]=target_box[:,0]*anchor[:,2]+anchor[:,0]
        boxes[:,1]=target_box[:,1]*anchor[:,3]+anchor[:,1]
        boxes[:,2]=np.exp(target_box[:,2])*anchor[:,2]
        boxes[:,3]=np.exp(target_box[:,3])*anchor[:,3]#[x,y,w,h]
        boxes=anchors_op.center_to_corner(boxes)
        for b in boxes:
            cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,0,0),2)

        cv2.rectangle(img,(int(gt_array[0]),int(gt_array[1])),(int(gt_array[2]),int(gt_array[3])),(0,0,255),1)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    #===========debug_all===============================
    coord.request_stop()
    coord.join(threads)