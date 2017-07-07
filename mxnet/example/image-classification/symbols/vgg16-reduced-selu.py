import mxnet as mx


def selu(data):
    alpha=1.6732632423543772848170429916717
    lda = 1.0507009873554804934193349852946

    tmp = mx.symbol.expm1(data)
    elu = alpha * tmp
    selu = mx.symbol.where(data >= 0, data, elu)
    selu = lda * selu
    return selu

def get_symbol(num_classes=1000, **kwargs):
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = selu(data=conv1_1)
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = selu(data=conv1_2)
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = selu(data=conv2_1)
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = selu(data=conv2_2)
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = selu(data=conv3_1)
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = selu(data=conv3_2)
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = selu(data=conv3_3)
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = selu(data=conv4_1)
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = selu(data=conv4_2)
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = selu(data=conv4_3)
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = selu(data=conv5_1)
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = selu(data=conv5_2)
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = selu(data=conv5_3)
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool5")
    # group 6
    conv6 = mx.symbol.Convolution(
        data=pool5, kernel=(3, 3), pad=(3, 3), dilate=(3, 3),
        num_filter=1024, name="conv6")
    relu6 = selu(data=conv6)
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="conv7")
    relu7 = selu(data=conv7)
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    conv8 = mx.symbol.Convolution(
        data=relu7, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="conv8")
    pool_last = mx.symbol.Pooling(data=conv8, kernel=(7,7), global_pool=True, name='pool_last', pool_type='avg')
    flatten = mx.sym.Flatten(data=pool_last, name="flatten")
    softmax = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')

    return softmax
