import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/python')
import mxnet as mx

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix='', no_bias=True, dw=0):
    Convolution = mx.symbol.Convolution
    if dw:
        Convolution = mx.symbol.ChannelwiseConvolution

    conv = Convolution(data=data, num_filter=num_filter, kernel=kernel,
            num_group=num_group, stride=stride, pad=pad, no_bias=no_bias, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def get_symbol(num_classes, **kwargs):
    data = mx.symbol.Variable(name="data")

    conv1 = Conv(data, num_filter=32, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv1')
    conv2_1_dw = Conv(conv1, num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv2_1_dw', num_group=32, dw=1)
    conv2_1 = Conv(conv2_1_dw, num_filter=64, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv2_1')
    conv2_2_dw = Conv(conv2_1, num_filter=64, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv2_2_dw', num_group=64, dw=1)
    conv2_2 = Conv(conv2_2_dw, num_filter=128, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv2_2')
    conv3_1_dw = Conv(conv2_2, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv3_1_dw', num_group=128, dw=1)
    conv3_1 = Conv(conv3_1_dw, num_filter=128, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv3_1')
    conv3_2_dw = Conv(conv3_1, num_filter=128, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv3_2_dw', num_group=128, dw=1)
    conv3_2 = Conv(conv3_2_dw, num_filter=256, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv3_2')
    conv4_1_dw = Conv(conv3_2, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv4_1_dw', num_group=256, dw=1)
    conv4_1 = Conv(conv4_1_dw, num_filter=256, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv4_1')
    conv4_2_dw = Conv(conv4_1, num_filter=256, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv4_2', num_group=256, dw=1)
    conv4_2 = Conv(conv4_2_dw, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv4_2_dw')
    conv5_1_dw = Conv(conv4_2, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_1_dw', num_group=512, dw=1)
    conv5_1 = Conv(conv5_1_dw, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv5_1')
    conv5_2_dw = Conv(conv5_1, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_2_dw', num_group=512, dw=1)
    conv5_2 = Conv(conv5_2_dw, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv5_2')
    conv5_3_dw = Conv(conv5_2, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_3_dw', num_group=512, dw=1)
    conv5_3 = Conv(conv5_3_dw, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv5_3')
    conv5_4_dw = Conv(conv5_3, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_4_dw', num_group=512, dw=1)
    conv5_4 = Conv(conv5_4_dw, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv5_4')
    conv5_5_dw = Conv(conv5_4, num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv5_5_dw', num_group=512, dw=1)
    conv5_5 = Conv(conv5_5_dw, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv5_5')
    conv5_6_dw = Conv(conv5_5, num_filter=512, kernel=(3,3), stride=(2,2), pad=(1,1), name='conv5_6_dw', num_group=512, dw=1)
    conv5_6 = Conv(conv5_6_dw, num_filter=1024, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv5_6')
    conv6_dw = Conv(conv5_6, num_filter=1024, kernel=(3,3), stride=(1,1), pad=(1,1), name='conv6_dw', num_group=1024, dw=1)
    conv6 = Conv(conv6_dw, num_filter=1024, kernel=(1,1), stride=(1,1), pad=(0,0), name='conv6')
    pool6 = mx.symbol.Pooling(conv6, kernel=(7,7), global_pool=True, pool_type='avg')
    flatten = mx.sym.Flatten(data=pool6, name="flatten")
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')

    return softmax


if __name__ == '__main__':
    sym = get_symbol(1000)
    import ipdb; ipdb.set_trace()
    internals = sym.get_internals()

    _, out_shapes, _ = internals.infer_shape(data=(1,3,224,224), softmax_label=(1,))

    sad = 0

