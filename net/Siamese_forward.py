from net.network import Network

class SiameseRPN(Network):
    def setup(self):
        (self.feed('img')
             #alex net layer 1-5
             .conv(11, 11, 96, 2, 2, padding='VALID', name='conv1')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 1, 1,padding='VALID', group=2, name='conv2')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 384, 1, 1, padding='VALID',name='conv3')
             .conv(3, 3, 384, 1, 1, padding='VALID',group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, padding='VALID',group=2, name='conv5'))
        (self.feed('conv5')
             #template
             .conv(3, 3, 2*self.k*256,1,1, padding='VALID', name='t_c')
             .reshape(rate=2,name='t_c_k'))
        (self.feed('conv5')
             #template
             .conv(3, 3, 4*self.k*256,1,1, padding='VALID', name='t_r')
             .reshape(rate=4,name='t_r_k'))
        (self.feed('conv5')
             #detection
             .conv(3, 3, 256,1,1, padding='VALID', name='d_c'))
        (self.feed('conv5')
             #detection
             .conv(3, 3, 256,1,1, padding='VALID', name='d_r'))
        (self.feed('conv_c','d_c')
             .cf_conv(padding='VALID', name='cls'))#[1,17,17,2k]
        (self.feed('conv_r','d_r')
             .cf_conv(padding='VALID', name='reg'))#[1,17,17,4k]