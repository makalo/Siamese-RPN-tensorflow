import numpy as np
from config import cfg
import cv2
from numba import jit
@jit
def debug(img,gt,pre_cls,pre_reg,pre_score,pre_box,label,target_box,step,anchor_op):
    img=(img*255).astype(np.uint8)
    #print('=============================================================')
    # pre_cls=pre_cls.reshape((-1,2))
    # pre_reg=pre_reg.reshape((-1,4))
    # print('===========box===========')
    # print(pre_box[np.where(label==1)])
    # print(target_box[np.where(label==1)])
    # print('===========box===========')
    # print('===========box===========')
    # print(pre_score[np.where(label==1)])
    # print('===========box===========')
    # print('===========cls===========')
    w = np.outer(np.hanning(17), np.hanning(17))
    w=np.stack([w,w,w,w,w],-1)
    w=w.reshape((-1))
    #w=np.tile(w.flatten(), 5)

    # index_cls=np.argmax(pre_cls[:,1])
    # boxes_reg=pre_reg[index_cls]
    # print('pre_cls_index={},pre_cls_max_value={},pre_reg={}'.format(index_cls,np.max(pre_cls[:,1]),boxes_reg))

    index_score=np.argsort(pre_score[:,1])[::-1][0:10]
    boxes_box=pre_box[index_score]
    #print('pre_score_index={},pre_score_max_value={},pre_box={}'.format(index_score,np.max(pre_score[:,1]*w),boxes_box))

    # should_score=pre_score[:,1]*label
    # index_should=np.argmax(should_score)
    # boxes_should=pre_box[index_should]
    # print('should_index={},should_max_value={},should_box={}'.format(index_should,np.max(should_score),boxes_should))

    index=index_score
    boxes=boxes_box
    #print(pre_score[:,1])
    #print('===========cls===========')


    # #============should_box===========
    # box_should=boxes_should
    # anchors_should=self.anchor_op.anchors
    # anchor_should=anchors_should[index_should]#[x1,y1,x2,y2]

    # anchor_should[2]=anchor_should[2]-anchor_should[0]
    # anchor_should[3]=anchor_should[3]-anchor_should[1]
    # anchor_should[0]=anchor_should[0]+(anchor_should[2])/2
    # anchor_should[1]=anchor_should[1]+(anchor_should[3])/2#[x,y,w,h]

    # b_should=np.zeros_like(box_should)
    # b_should[0]=box_should[0]*anchor_should[2]+anchor_should[0]
    # b_should[1]=box_should[1]*anchor_should[3]+anchor_should[1]
    # b_should[2]=np.exp(box_should[2])*anchor_should[2]
    # b_should[3]=np.exp(box_should[3])*anchor_should[3]#[x,y,w,h]

    # b_should[0]=b_should[0]-b_should[2]/2
    # b_should[1]=b_should[1]-b_should[3]/2
    # b_should[2]=b_should[0]+b_should[2]
    # b_should[3]=b_should[1]+b_should[3]#[x1,y1,x2,y2]

    # if b_should[2]<1000 and b_should[3]<1000:
    #     cv2.rectangle(img,(int(b_should[0]),int(b_should[1])),(int(b_should[2]),int(b_should[3])),(0,255,0),1)
    # #============should_box===========


    #============pre_box===========
    box=boxes
    anchors=anchor_op.regu()
    anchors=anchor_op.corner_to_center(anchors)
    diff_anchors=anchor_op.diff_anchor_gt(gt,anchors)
    anchor=anchors[index]#[x1,y1,x2,y2]


    # anchor[2]=anchor[2]-anchor[0]
    # anchor[3]=anchor[3]-anchor[1]
    # anchor[0]=anchor[0]+(anchor[2])/2
    # anchor[1]=anchor[1]+(anchor[3])/2#[x,y,w,h]

    b=np.zeros_like(box)
    b[:,0]=box[:,0]*anchor[:,2]+anchor[:,0]
    b[:,1]=box[:,1]*anchor[:,3]+anchor[:,1]
    b[:,2]=np.exp(box[:,2])*anchor[:,2]
    b[:,3]=np.exp(box[:,3])*anchor[:,3]#[x,y,w,h]

    # b[0]=b[0]-b[2]/2
    # b[1]=b[1]-b[3]/2
    # b[2]=b[0]+b[2]
    # b[3]=b[1]+b[3]#[x1,y1,x2,y2]
    b=anchor_op.center_to_corner(b)
    anchor=anchor_op.center_to_corner(anchor)


    #if b[2]<1000 and b[3]<1000:
    for bbox in b:
        color = np.random.random((3, )) * 0.6 + 0.4
        color = color * 255
        color = color.astype(np.int32).tolist()
        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),color,1)
    cv2.rectangle(img,(int(b[0][0]),int(b[0][1])),(int(b[0][2]),int(b[0][3])),(0,0,0),2)

    cv2.rectangle(img,(int(anchor[0][0]),int(anchor[0][1])),(int(anchor[0][2]),int(anchor[0][3])),(255,0,0),1)
    #============pre_box===========



    #============gt_box===========
    # gt_b=np.zeros_like(box)
    # gt_b[0]=(gt[0]-anchor[0])/(anchor[2]+0.01)
    # gt_b[1]=(gt[1]-anchor[1])/(anchor[3]+0.01)
    # gt_b[2]=np.log(gt[2]/(anchor[2]+0.01))
    # gt_b[3]=np.log(gt[3]/(anchor[3]+0.01))
    # print('++++++offset+++++++')
    # print('pre={}'.format(box))
    # print('comput_tg={}'.format(gt_b))
    # print('target={}'.format(target_box[index]))
    # print('anchor={}'.format(anchor))
    # print('++++++offset+++++++')

    gt[0]=gt[0]-gt[2]/2
    gt[1]=gt[1]-gt[3]/2
    gt[2]=gt[0]+gt[2]
    gt[3]=gt[1]+gt[3]
    cv2.rectangle(img,(int(gt[0]),int(gt[1])),(int(gt[2]),int(gt[3])),(0,0,255),2)
    cv2.imwrite(cfg.debug_dir+'/'+str(step)+'.jpg',img)
    #============gt_box===========
    # print('===========reg===========')
    # print(b.astype(np.int32))
    # print(np.array(gt).astype(np.int32))
    # print('===========reg===========')
    # print('====================================================')