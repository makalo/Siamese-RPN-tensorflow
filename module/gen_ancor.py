import numpy as np
import cv2
class Anchor():
    """docstring for Anchor"""
    def __init__(self,feature_w,feature_h):
        self.width=255
        self.height=255
        self.w=feature_w
        self.h=feature_h
        self.base=64
        self.stride=16
        self.scale=[1/3,1/2,1,2,3]
        self.anchors=self.gen_anchors()
    def gen_single_anchor(self):
        scale=np.array(self.scale)
        s=self.base*self.base
        w=np.sqrt(s/scale)
        h=w*scale
        c_x=(self.stride-1)/2
        c_y=(self.stride-1)/2
        anchor=np.vstack([c_x*np.ones_like(scale),c_y*np.ones_like(scale),w,h])
        anchor=anchor.transpose()#[x,y,w,h]
        anchor=self.center_to_corner(anchor)#[x1,y1,x2,y2]
        anchor=anchor.astype(np.int32)
        return anchor
    def gen_anchors(self):
        anchor=self.gen_single_anchor()
        k=anchor.shape[0]
        shift_x=[x*self.stride for x in range(self.w)]
        shift_y=[y*self.stride for y in range(self.h)]
        shift_x,shift_y=np.meshgrid(shift_x,shift_y)
        shifts=np.vstack([shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel()]).transpose()
        a=shifts.shape[0]
        anchors=anchor.reshape((1,k,4))+shifts.reshape((a,1,4))
        anchors=anchors.reshape((a*k,4))#[x1,y1,x2,y2]
        #anchors=self.corner_to_center(anchors)#[x,y,w,h]
        anchors=anchors.astype(np.float32)
        return anchors
    def center_to_corner(self,box):
        box_temp=np.zeros_like(box)
        box_temp[:,0]=box[:,0]-(box[:,2]-1)/2
        box_temp[:,1]=box[:,1]-(box[:,3]-1)/2
        box_temp[:,2]=box[:,0]+(box[:,2]-1)/2
        box_temp[:,3]=box[:,1]+(box[:,3]-1)/2
        #box_temp=box_temp.astype(np.int32)
        return box_temp
    def corner_to_center(self,box):
        box_temp=np.zeros_like(box)
        box_temp[:,0]=box[:,0]+(box[:,2]-box[:,0])/2
        box_temp[:,1]=box[:,1]+(box[:,3]-box[:,1])/2
        box_temp[:,2]=(box[:,2]-box[:,0])
        box_temp[:,3]=(box[:,3]-box[:,1])
        #box_temp=box_temp.astype(np.int32)
        return box_temp



    def diff_anchor_gt(self,gt,anchors):
        #gt [x,y,w,h]
        #anchors [x,y,w,h]
        diff_anchors=np.zeros_like(anchors).astype(np.float32)
        diff_anchors[:,0]=(gt[0]-anchors[:,0])/(anchors[:,2]+0.01)
        diff_anchors[:,1]=(gt[1]-anchors[:,1])/(anchors[:,3]+0.01)
        diff_anchors[:,2]=np.log(gt[2]/(anchors[:,2]+0.01))
        diff_anchors[:,3]=np.log(gt[3]/(anchors[:,3]+0.01))
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
        N=box1.shape[0]
        K=box2.shape[0]
        box1=np.array(box1.reshape((N,1,4)))+np.zeros((1,K,4))#box1=[N,K,4]
        box2=np.array(box2.reshape((1,K,4)))+np.zeros((N,1,4))#box1=[N,K,4]
        x_max=np.max(np.stack((box1[:,:,0],box2[:,:,0]),axis=-1),axis=2)
        x_min=np.min(np.stack((box1[:,:,2],box2[:,:,2]),axis=-1),axis=2)
        y_max=np.max(np.stack((box1[:,:,1],box2[:,:,1]),axis=-1),axis=2)
        y_min=np.min(np.stack((box1[:,:,3],box2[:,:,3]),axis=-1),axis=2)
        tb=x_min-x_max
        lr=y_min-y_max
        tb[np.where(tb<0)]=0
        lr[np.where(lr<0)]=0
        over_square=tb*lr
        all_square=(box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])+(box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])-over_square
        return over_square/all_square

    def pos_neg_anchor(self,gt):
        inds_inside = np.where(
        (self.anchors[:, 0] >= 0) &
        (self.anchors[:, 1] >= 0) &
        (self.anchors[:, 2] < self.width) &  # width
        (self.anchors[:, 3] < self.height)    # height
        )[0]
        all_box=np.zeros((self.anchors.shape[0],4),dtype=np.float32)
        target_box=np.zeros((self.anchors.shape[0],4),dtype=np.float32)
        target_inside_weight_box=np.zeros((self.anchors.shape[0],4),dtype=np.float32)
        target_outside_weight_box=np.ones((self.anchors.shape[0],4),dtype=np.float32)
        label=-np.ones((self.anchors.shape[0],),dtype=np.float32)


        anchors_inside=(self.anchors[inds_inside]).astype(np.float32)
        mask_label_inside=-np.ones((len(inds_inside),),dtype=np.float32)

        gt_array=np.array(gt).reshape((1,4))
        gt_array=self.center_to_corner(gt_array)


        iou_value=self.iou(anchors_inside,gt_array)

        pos=np.zeros_like(iou_value)
        pos[np.where(iou_value>0.3)]=iou_value[np.where(iou_value>0.3)]
        pos_index=np.argsort(pos[:,0])[::-1]
        pos_num=min(len(pos_index),16)
        pos_index=pos_index[:pos_num]
        mask_label_inside[pos_index]=1

        neg=np.zeros_like(iou_value)
        #neg[np.where(iou_value<0.3)]=iou_value[np.where(iou_value<0.3)]
        #neg_index=np.argsort(neg[:,0])[::-1]
        neg_index=np.where(iou_value<0.3)[0]
        #print(neg_index)
        neg_index=np.random.choice(neg_index,(64-pos_num))
        #print(neg_index)

        # neg_num=min(len(neg_index),64-pos_num)
        # neg_index=neg_index[:neg_num]
        mask_label_inside[neg_index]=0

        diff_anchors=self.diff_anchor_gt(gt,self.corner_to_center(anchors_inside))
        #print(diff_anchors.shape)
        target_box[inds_inside]=diff_anchors

        all_box[inds_inside]=anchors_inside
        label[inds_inside]=mask_label_inside

        target_inside_weight_box[np.where(label==1)]=np.array([1.,1.,1.,1.])
        target_outside_weight_box=target_outside_weight_box*1.0/len(np.where(label==1)[0])
        #print(target_outside_weight_box[np.where(target_outside_weight_box>0)])
        return label,target_box,target_inside_weight_box,target_outside_weight_box,all_box

    def pos_neg_anchor2(self,gt):
        all_box=self.anchors.copy()
        all_box[np.where(all_box<0)]=0
        all_box[np.where(all_box>self.width-1)]=self.width-1
        target_box=np.zeros((self.anchors.shape[0],4),dtype=np.float32)
        target_inside_weight_box=np.zeros((self.anchors.shape[0],4),dtype=np.float32)
        target_outside_weight_box=np.ones((self.anchors.shape[0],4),dtype=np.float32)
        label=-np.ones((self.anchors.shape[0],),dtype=np.float32)

        gt_array=np.array(gt).reshape((1,4))
        gt_array=self.center_to_corner(gt_array)

        iou_value=self.iou(all_box,gt_array)

        pos=np.zeros_like(iou_value)
        pos[np.where(iou_value>0.3)]=iou_value[np.where(iou_value>0.3)]
        pos_index=np.argsort(pos[:,0])[::-1]
        pos_num=min(len(pos_index),16)
        pos_index=pos_index[:pos_num]
        label[pos_index]=1

        neg=np.zeros_like(iou_value)
        neg_index=np.where(iou_value<0.3)[0]
        neg_index=np.random.choice(neg_index,(64-pos_num))
        # neg_num=min(len(neg_index),64-pos_num)
        # neg_index=neg_index[:neg_num]
        label[neg_index]=0

        target_box=self.diff_anchor_gt(gt,self.corner_to_center(all_box))
        target_inside_weight_box[np.where(label==1)]=np.array([1.,1.,1.,1.])
        target_outside_weight_box=target_outside_weight_box*1.0/len(np.where(label==1)[0])
        #print(target_outside_weight_box[np.where(target_outside_weight_box>0)])
        return label,target_box,target_inside_weight_box,target_outside_weight_box,all_box
    def regu(self):
        all_box=self.anchors.copy()
        all_box[np.where(all_box<0)]=0
        all_box[np.where(all_box>self.width-1)]=self.width-1
        return all_box


if __name__=='__main__':
    import sys
    sys.path.append('../')
    from utils.image_reader import Image_reader
    reader=Image_reader('../data/vot2013')
    template_p,template_label_p,detection_p,detection_label_p,offset,ratio,detection,detection_label=reader.get_data()
    test=Anchor(17,17)
    img=np.ones((255,255,3),dtype=np.uint8)*255
    img=(detection_p*255).astype(np.uint8)
#===========debug_all===============================
    # color = np.random.random((3, )) * 0.6 + 0.4
    # color = color * 255
    # color = color.astype(np.int32).tolist()
    #gt=[100,100,50,50]
    gt=detection_label_p
    gt_array=np.array(gt).reshape((1,4))
    gt_array=test.center_to_corner(gt_array)[0]
    label,target_box,_,_,all_box=test.pos_neg_anchor2(gt)

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

    cv2.rectangle(img,(int(gt_array[0]),int(gt_array[1])),(int(gt_array[2]),int(gt_array[3])),(0,0,255),1)
    cv2.imshow('img',img)
    cv2.waitKey(0)
#===========debug_all===============================



# #===========debug_single_anchor===============================
#     all_anchors=test.anchors.reshape((-1,5,4))
#     box=all_anchors[143,:,:]
#     for b in box:
#         cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,255,0),1)
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
# #===========debug_single_anchor===============================



# #===========debug_ahchors===============================
#     all_anchors=test.anchors.reshape((-1,5,4))
#     box=all_anchors[:,2,:]
#     for b in box:
#         cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(0,255,0),1)
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
# #===========debug_ahchors===============================




# #===========debug_iou===============================
#     box1=np.array([0,0,50,50]).reshape((1,4))
#     box2=np.array([0,0,100,100]).reshape((1,4))
#     print(test.iou(box1,box2))
# #===========debug_iou===============================