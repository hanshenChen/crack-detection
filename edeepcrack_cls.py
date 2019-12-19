import tensorflow as tf
from tensorflow import keras
import datetime as dt
import numpy as np
import tensorflow as tf
from indices_pooling import MaxUnpooling2D,MaxPoolingWithArgmax2D

tf.enable_eager_execution()

def crop(variable, th, tw):
    h, w = variable.shape._dims[1].value, variable.shape._dims[2].value
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    #print(h,w,x1,y1,x1+tw,y1+th)
    return tf.keras.layers.Lambda(lambda x: tf.slice(x, (0,y1,x1,0), (-1,th,tw,-1)))(variable)

    #return K.slice(variable, (0,y1,x1,0), (-1,th,tw,-1))#variable[:, :, y1 : y1 + th, x1 : x1 + tw]


class Skip_layers(tf.keras.Model):

    def __init__(self, scale, img_H, img_W):
        super(Skip_layers, self).__init__()

        self.scale = scale
        self.img_H = img_H
        self.img_W = img_W

        self.conv1=keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',name="sk_conv_"+ str(scale),padding='valid')
        self.bn1  =keras.layers.BatchNormalization()
        self.convtran1=keras.layers.Conv2DTranspose(filters=1, kernel_size=(2 * scale, 2 * scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.crop = crop()
        self.activation1 = keras.layers.Activation(activation='sigmoid')

    def call(self, endata, dedata):
        x = tf.concat([endata, dedata], axis=3)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.scale != 1:
            x= self.convtran1(x)
        x = crop(x,self.img_H,self.img_W)
        #x = self.activation1(x)
        return x

class Deepcrack(tf.keras.Model):
    def __init__(self,input_shape):
        super(Deepcrack, self).__init__()

        self.img_H = input_shape[1]
        self.img_W = input_shape[2]

        self.block1_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="block1_conv1",padding='same')
        self.block1_bn1  =keras.layers.BatchNormalization(name="block1_bn1")
        self.block1_relu1=keras.layers.Activation('relu')
        self.block1_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="block1_conv2",padding="same")
        self.block1_bn2  =keras.layers.BatchNormalization(name="block1_bn2")
        self.block1_relu2=keras.layers.Activation("relu")        
        self.pool1 = MaxPoolingWithArgmax2D((2, 2), name='block1_pool')

        self.block2_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="block2_conv1",padding='same')
        self.block2_bn1  =keras.layers.BatchNormalization(name="block2_bn1")
        self.block2_relu1=keras.layers.Activation('relu')
        self.block2_conv2=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="block2_conv2",padding="same")
        self.block2_bn2  =keras.layers.BatchNormalization(name="block2_bn2")
        self.block2_relu2=keras.layers.Activation("relu")
        self.pool2 = MaxPoolingWithArgmax2D((2, 2), name='block2_pool')

        self.block3_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="block3_conv1",padding='same')
        self.block3_bn1  =keras.layers.BatchNormalization(name="block3_bn1")
        self.block3_relu1=keras.layers.Activation('relu')
        self.block3_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="block3_conv2",padding="same")
        self.block3_bn2  =keras.layers.BatchNormalization(name="block3_bn2")
        self.block3_relu2=keras.layers.Activation("relu")
        self.block3_conv3=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="block3_conv3",padding="same")
        self.block3_bn3  =keras.layers.BatchNormalization(name="block3_bn3")
        self.block3_relu3=keras.layers.Activation("relu")
        self.pool3 = MaxPoolingWithArgmax2D((2, 2), name='block3_pool')

        self.block4_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="block4_conv1",padding='same')
        self.block4_bn1  =keras.layers.BatchNormalization(name="block4_bn1")
        self.block4_relu1=keras.layers.Activation('relu')
        self.block4_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block4_conv2",padding="same")
        self.block4_bn2  =keras.layers.BatchNormalization(name="block4_bn2")
        self.block4_relu2=keras.layers.Activation("relu")
        self.block4_conv3=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block4_conv3",padding="same")
        self.block4_bn3  =keras.layers.BatchNormalization(name="block4_bn3")
        self.block4_relu3=keras.layers.Activation("relu")
        self.pool4 = MaxPoolingWithArgmax2D((2, 2), name='block4_pool')

        self.block5_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="block5_conv1",padding='same')
        self.block5_bn1  =keras.layers.BatchNormalization(name="block5_bn1")
        self.block5_relu1=keras.layers.Activation('relu')
        self.block5_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block5_conv2",padding="same")
        self.block5_bn2  =keras.layers.BatchNormalization(name="block5_bn2")
        self.block5_relu2=keras.layers.Activation("relu")
        self.block5_conv3=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block5_conv3",padding="same")
        self.block5_bn3  =keras.layers.BatchNormalization(name="block5_bn3")
        self.block5_relu3=keras.layers.Activation("relu")
        self.pool5 = MaxPoolingWithArgmax2D((2, 2), name='block5_pool')

        self.up1 = MaxUnpooling2D(up_size = (2,2))
        self.up1_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="up1_conv1",padding='same')
        self.up1_bn1  =keras.layers.BatchNormalization(name="up1_bn1")
        self.up1_relu1=keras.layers.Activation('relu')
        self.up1_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="up1_conv2",padding="same")
        self.up1_bn2  =keras.layers.BatchNormalization(name="up1_bn2")
        self.up1_relu2=keras.layers.Activation("relu")
        self.up1_conv3=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="up1_conv3",padding="same")
        self.up1_bn3  =keras.layers.BatchNormalization(name="up1_bn3")
        self.up1_relu3=keras.layers.Activation("relu")

        self.up2 = MaxUnpooling2D(up_size = (2,2))
        self.up2_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="up2_conv1",padding='same')
        self.up2_bn1  =keras.layers.BatchNormalization(name="up2_bn1")
        self.up2_relu1=keras.layers.Activation('relu')
        self.up2_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="up2_conv2",padding="same")
        self.up2_bn2  =keras.layers.BatchNormalization(name="up2_bn2")
        self.up2_relu2=keras.layers.Activation("relu")
        self.up2_conv3=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="up2_conv3",padding="same")
        self.up2_bn3  =keras.layers.BatchNormalization(name="up2_bn3")
        self.up2_relu3=keras.layers.Activation("relu")

        self.up3 = MaxUnpooling2D(up_size = (2,2))
        self.up3_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="up3_conv1",padding='same')
        self.up3_bn1  =keras.layers.BatchNormalization(name="up3_bn1")
        self.up3_relu1=keras.layers.Activation('relu')
        self.up3_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="up3_conv2",padding="same")
        self.up3_bn2  =keras.layers.BatchNormalization(name="up3_bn2")
        self.up3_relu2=keras.layers.Activation("relu")
        self.up3_conv3=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="up3_conv3",padding="same")
        self.up3_bn3  =keras.layers.BatchNormalization(name="up3_bn3")
        self.up3_relu3=keras.layers.Activation("relu")


        self.up4 = MaxUnpooling2D(up_size = (2,2))
        self.up4_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="up4_conv1",padding='same')
        self.up4_bn1  =keras.layers.BatchNormalization(name="up4_bn1")
        self.up4_relu1=keras.layers.Activation('relu')
        self.up4_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="up4_conv2",padding="same")
        self.up4_bn2  =keras.layers.BatchNormalization(name="up4_bn2")
        self.up4_relu2=keras.layers.Activation("relu")

        self.up5 = MaxUnpooling2D(up_size = (2,2))
        self.up5_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="up5_conv1",padding='same')
        self.up5_bn1  =keras.layers.BatchNormalization(name="up5_bn1")
        self.up5_relu1=keras.layers.Activation('relu')
        self.up5_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="up5_conv2",padding="same")
        self.up5_bn2  =keras.layers.BatchNormalization(name="up5_bn2")
        self.up5_relu2=keras.layers.Activation("relu")

        scale=1
        self.sk1_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1,1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk1_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        #self.sk1_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
        #                                              padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk1_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=2
        self.sk2_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk2_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk2_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk2_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=4
        self.sk3_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk3_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk3_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk3_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=8
        self.sk4_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk4_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk4_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk4_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=16
        self.sk5_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk5_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk5_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk5_activation1 = keras.layers.Activation(activation='sigmoid')

        self.conv10 = keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid', name="c1", kernel_initializer='he_normal')
        self.build(input_shape)
        self.black=np.zeros((1,input_shape[1],input_shape[2],1))

    def call(self, inputs):
        #print(tf.executing_eagerly())
        x = self.block1_conv1(inputs)
        x = self.block1_bn1(x)
        x = self.block1_relu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        block1 = self.block1_relu2(x)
        pool1,mask1 = self.pool1(block1)

        x = self.block2_conv1(pool1)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        block2 = self.block2_relu2(x)
        pool2,mask2 = self.pool2(block2)

        x = self.block3_conv1(pool2)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_conv3(x)
        x = self.block3_bn3(x)
        block3 = self.block3_relu3(x)
        pool3,mask3 = self.pool3(block3)

        x = self.block4_conv1(pool3)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_conv3(x)
        x = self.block4_bn3(x)
        block4 = self.block4_relu3(x)
        pool4,mask4 = self.pool4(block4)

        x = self.block5_conv1(pool4)
        x = self.block5_bn1(x)
        x = self.block5_relu1(x)
        x = self.block5_conv2(x)
        x = self.block5_bn2(x)
        x = self.block5_relu2(x)
        x = self.block5_conv3(x)
        x = self.block5_bn3(x)
        block5 = self.block5_relu3(x)
        pool5,mask5 = self.pool5(block5)

        x = self.up1([pool5,mask5])
        x = self.up1_conv1(x)
        x = self.up1_bn1(x)
        x = self.up1_relu1(x)
        x = self.up1_conv2(x)
        x = self.up1_bn2(x)
        x = self.up1_relu2(x)
        x = self.up1_conv3(x)
        x = self.up1_bn3(x)
        up1 = self.up1_relu3(x)

        x = self.up2([up1,mask4])
        x = self.up2_conv1(x)
        x = self.up2_bn1(x)
        x = self.up2_relu1(x)
        x = self.up2_conv2(x)
        x = self.up2_bn2(x)
        x = self.up2_relu2(x)
        x = self.up2_conv3(x)
        x = self.up2_bn3(x)
        up2 = self.up2_relu3(x)

        x = self.up3([up2,mask3])
        x = self.up3_conv1(x)
        x = self.up3_bn1(x)
        x = self.up3_relu1(x)
        x = self.up3_conv2(x)
        x = self.up3_bn2(x)
        x = self.up3_relu2(x)
        x = self.up3_conv3(x)
        x = self.up3_bn3(x)
        up3 = self.up3_relu3(x)


        x = self.up4([up3,mask2])
        x = self.up4_conv1(x)
        x = self.up4_bn1(x)
        x = self.up4_relu1(x)
        x = self.up4_conv2(x)
        x = self.up4_bn2(x)
        up4 = self.up4_relu2(x)


        x = self.up5([up4,mask1])
        x = self.up5_conv1(x)
        x = self.up5_bn1(x)
        x = self.up5_relu1(x)
        x = self.up5_conv2(x)
        x = self.up5_bn2(x)
        up5 = self.up5_relu2(x)

        x = tf.concat([block1, up5], axis=3)
        x = self.sk1_conv1(x)
        x = self.sk1_bn1(x)
        deconv1 = crop(x, self.img_H, self.img_W)

        x = tf.concat([block2, up4], axis=3)
        x = self.sk2_conv1(x)
        x = self.sk2_bn1(x)
        x = self.sk2_convtran1(x)
        deconv2 = crop(x, self.img_H, self.img_W)

        x = tf.concat([block3, up3], axis=3)
        x = self.sk3_conv1(x)
        x = self.sk3_bn1(x)
        x = self.sk3_convtran1(x)
        deconv3 = crop(x, self.img_H, self.img_W)

        x = tf.concat([block4, up2], axis=3)
        x = self.sk4_conv1(x)
        x = self.sk4_bn1(x)
        x = self.sk4_convtran1(x)
        deconv4 = crop(x, self.img_H, self.img_W)

        x = tf.concat([block5, up1], axis=3)
        x = self.sk5_conv1(x)
        x = self.sk5_bn1(x)
        x = self.sk5_convtran1(x)
        deconv5 = crop(x, self.img_H, self.img_W)

        mergeall = tf.concat([deconv1, deconv2, deconv3, deconv4, deconv5], axis=-1)

        output = self.conv10(mergeall)

        return output


class Deepcrack_cls(tf.keras.Model):
    def __init__(self,input_shape):
        super(Deepcrack_cls, self).__init__()

        self.img_H = input_shape[1]
        self.img_W = input_shape[2]

        self.block1_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="block1_conv1",padding='same')
        self.block1_bn1  =keras.layers.BatchNormalization(name="block1_bn1")
        self.block1_relu1=keras.layers.Activation('relu')
        self.block1_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="block1_conv2",padding="same")
        self.block1_bn2  =keras.layers.BatchNormalization(name="block1_bn2")
        self.block1_relu2=keras.layers.Activation("relu")
        self.pool1 = MaxPoolingWithArgmax2D((2, 2), name='block1_pool')

        self.block2_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="block2_conv1",padding='same')
        self.block2_bn1  =keras.layers.BatchNormalization(name="block2_bn1")
        self.block2_relu1=keras.layers.Activation('relu')
        self.block2_conv2=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="block2_conv2",padding="same")
        self.block2_bn2  =keras.layers.BatchNormalization(name="block2_bn2")
        self.block2_relu2=keras.layers.Activation("relu")
        self.pool2 = MaxPoolingWithArgmax2D((2, 2), name='block2_pool')

        self.block3_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="block3_conv1",padding='same')
        self.block3_bn1  =keras.layers.BatchNormalization(name="block3_bn1")
        self.block3_relu1=keras.layers.Activation('relu')
        self.block3_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="block3_conv2",padding="same")
        self.block3_bn2  =keras.layers.BatchNormalization(name="block3_bn2")
        self.block3_relu2=keras.layers.Activation("relu")
        self.block3_conv3=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="block3_conv3",padding="same")
        self.block3_bn3  =keras.layers.BatchNormalization(name="block3_bn3")
        self.block3_relu3=keras.layers.Activation("relu")
        self.pool3 = MaxPoolingWithArgmax2D((2, 2), name='block3_pool')

        self.block4_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="block4_conv1",padding='same')
        self.block4_bn1  =keras.layers.BatchNormalization(name="block4_bn1")
        self.block4_relu1=keras.layers.Activation('relu')
        self.block4_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block4_conv2",padding="same")
        self.block4_bn2  =keras.layers.BatchNormalization(name="block4_bn2")
        self.block4_relu2=keras.layers.Activation("relu")
        self.block4_conv3=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block4_conv3",padding="same")
        self.block4_bn3  =keras.layers.BatchNormalization(name="block4_bn3")
        self.block4_relu3=keras.layers.Activation("relu")
        self.pool4 = MaxPoolingWithArgmax2D((2, 2), name='block4_pool')

        self.block5_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="block5_conv1",padding='same')
        self.block5_bn1  =keras.layers.BatchNormalization(name="block5_bn1")
        self.block5_relu1=keras.layers.Activation('relu')
        self.block5_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block5_conv2",padding="same")
        self.block5_bn2  =keras.layers.BatchNormalization(name="block5_bn2")
        self.block5_relu2=keras.layers.Activation("relu")
        self.block5_conv3=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block5_conv3",padding="same")
        self.block5_bn3  =keras.layers.BatchNormalization(name="block5_bn3")
        self.block5_relu3=keras.layers.Activation("relu")
        self.pool5 = MaxPoolingWithArgmax2D((2, 2), name='block5_pool')

        self.up1 = MaxUnpooling2D(up_size = (2,2))
        self.up1_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="up1_conv1",padding='same')
        self.up1_bn1  =keras.layers.BatchNormalization(name="up1_bn1")
        self.up1_relu1=keras.layers.Activation('relu')
        self.up1_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="up1_conv2",padding="same")
        self.up1_bn2  =keras.layers.BatchNormalization(name="up1_bn2")
        self.up1_relu2=keras.layers.Activation("relu")
        self.up1_conv3=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="up1_conv3",padding="same")
        self.up1_bn3  =keras.layers.BatchNormalization(name="up1_bn3")
        self.up1_relu3=keras.layers.Activation("relu")

        self.up2 = MaxUnpooling2D(up_size = (2,2))
        self.up2_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="up2_conv1",padding='same')
        self.up2_bn1  =keras.layers.BatchNormalization(name="up2_bn1")
        self.up2_relu1=keras.layers.Activation('relu')
        self.up2_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="up2_conv2",padding="same")
        self.up2_bn2  =keras.layers.BatchNormalization(name="up2_bn2")
        self.up2_relu2=keras.layers.Activation("relu")
        self.up2_conv3=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="up2_conv3",padding="same")
        self.up2_bn3  =keras.layers.BatchNormalization(name="up2_bn3")
        self.up2_relu3=keras.layers.Activation("relu")

        self.up3 = MaxUnpooling2D(up_size = (2,2))
        self.up3_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="up3_conv1",padding='same')
        self.up3_bn1  =keras.layers.BatchNormalization(name="up3_bn1")
        self.up3_relu1=keras.layers.Activation('relu')
        self.up3_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="up3_conv2",padding="same")
        self.up3_bn2  =keras.layers.BatchNormalization(name="up3_bn2")
        self.up3_relu2=keras.layers.Activation("relu")
        self.up3_conv3=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="up3_conv3",padding="same")
        self.up3_bn3  =keras.layers.BatchNormalization(name="up3_bn3")
        self.up3_relu3=keras.layers.Activation("relu")


        self.up4 = MaxUnpooling2D(up_size = (2,2))
        self.up4_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="up4_conv1",padding='same')
        self.up4_bn1  =keras.layers.BatchNormalization(name="up4_bn1")
        self.up4_relu1=keras.layers.Activation('relu')
        self.up4_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="up4_conv2",padding="same")
        self.up4_bn2  =keras.layers.BatchNormalization(name="up4_bn2")
        self.up4_relu2=keras.layers.Activation("relu")

        self.up5 = MaxUnpooling2D(up_size = (2,2))
        self.up5_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="up5_conv1",padding='same')
        self.up5_bn1  =keras.layers.BatchNormalization(name="up5_bn1")
        self.up5_relu1=keras.layers.Activation('relu')
        self.up5_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="up5_conv2",padding="same")
        self.up5_bn2  =keras.layers.BatchNormalization(name="up5_bn2")
        self.up5_relu2=keras.layers.Activation("relu")

        scale=1
        self.sk1_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1,1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk1_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        #self.sk1_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
        #                                              padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk1_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=2
        self.sk2_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk2_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk2_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk2_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=4
        self.sk3_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk3_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk3_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk3_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=8
        self.sk4_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk4_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk4_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk4_activation1 = keras.layers.Activation(activation='sigmoid')

        scale=16
        self.sk5_conv1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), kernel_initializer='he_normal',
                                         name="sk_conv" + str(scale), padding='valid')
        self.sk5_bn1 = keras.layers.BatchNormalization(name="sk_bn" + str(scale))
        self.sk5_convtran1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(2*scale, 2*scale), strides=scale,
                                                      padding='valid', activation=None, name="sk_deconv" + str(scale))
        #self.sk5_activation1 = keras.layers.Activation(activation='sigmoid')

        self.conv10 = keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid', name="c1", kernel_initializer='he_normal')

        self.cls_max1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='cblock5_pool3')
        self.cls_conv1 = keras.layers.Conv2D(256, (3, 3), padding='same', name='cblock5_conv3')
        self.cls_bn1  = keras.layers.BatchNormalization(name='cblock5_bn3')
        self.cls_relu1 = keras.layers.Activation('relu')
        self.cls_conv2 = keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same',name='sfconv2')
        self.cls_gmax2 = keras.layers.GlobalMaxPooling2D(name='gmax_pool1')


        self.build(input_shape)
        self.black=np.zeros((1,input_shape[1],input_shape[2],1))

    def call(self, inputs):

        #print(tf.executing_eagerly())
        x = self.block1_conv1(inputs)
        x = self.block1_bn1(x)
        x = self.block1_relu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        block1 = self.block1_relu2(x)
        pool1,mask1 = self.pool1(block1)

        x = self.block2_conv1(pool1)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        block2 = self.block2_relu2(x)
        pool2,mask2 = self.pool2(block2)

        x = self.block3_conv1(pool2)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_conv3(x)
        x = self.block3_bn3(x)
        block3 = self.block3_relu3(x)
        pool3,mask3 = self.pool3(block3)

        x = self.block4_conv1(pool3)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_conv3(x)
        x = self.block4_bn3(x)
        block4 = self.block4_relu3(x)
        pool4,mask4 = self.pool4(block4)

        x = self.block5_conv1(pool4)
        x = self.block5_bn1(x)
        x = self.block5_relu1(x)
        x = self.block5_conv2(x)
        x = self.block5_bn2(x)
        x = self.block5_relu2(x)
        x = self.block5_conv3(x)
        x = self.block5_bn3(x)
        block5 = self.block5_relu3(x)
        pool5,mask5 = self.pool5(block5)

        c=self.cls_max1(block5)
        c=self.cls_conv1(c)
        c=self.cls_bn1(c)
        c=self.cls_relu1(c)
        c=self.cls_conv2(c)
        c=self.cls_gmax2(c)

        if tf.executing_eagerly()==False:
            print("not executing_eagerly")#
            x = self.up1([pool5,mask5])
            x = self.up1_conv1(x)
            x = self.up1_bn1(x)
            x = self.up1_relu1(x)
            x = self.up1_conv2(x)
            x = self.up1_bn2(x)
            x = self.up1_relu2(x)
            x = self.up1_conv3(x)
            x = self.up1_bn3(x)
            up1 = self.up1_relu3(x)

            x = self.up2([up1,mask4])
            x = self.up2_conv1(x)
            x = self.up2_bn1(x)
            x = self.up2_relu1(x)
            x = self.up2_conv2(x)
            x = self.up2_bn2(x)
            x = self.up2_relu2(x)
            x = self.up2_conv3(x)
            x = self.up2_bn3(x)
            up2 = self.up2_relu3(x)

            x = self.up3([up2,mask3])
            x = self.up3_conv1(x)
            x = self.up3_bn1(x)
            x = self.up3_relu1(x)
            x = self.up3_conv2(x)
            x = self.up3_bn2(x)
            x = self.up3_relu2(x)
            x = self.up3_conv3(x)
            x = self.up3_bn3(x)
            up3 = self.up3_relu3(x)


            x = self.up4([up3,mask2])
            x = self.up4_conv1(x)
            x = self.up4_bn1(x)
            x = self.up4_relu1(x)
            x = self.up4_conv2(x)
            x = self.up4_bn2(x)
            up4 = self.up4_relu2(x)


            x = self.up5([up4,mask1])
            x = self.up5_conv1(x)
            x = self.up5_bn1(x)
            x = self.up5_relu1(x)
            x = self.up5_conv2(x)
            x = self.up5_bn2(x)
            up5 = self.up5_relu2(x)

            x = tf.concat([block1, up5], axis=3)
            x = self.sk1_conv1(x)
            x = self.sk1_bn1(x)
            deconv1 = crop(x, self.img_H, self.img_W)

            x = tf.concat([block2, up4], axis=3)
            x = self.sk2_conv1(x)
            x = self.sk2_bn1(x)
            x = self.sk2_convtran1(x)
            deconv2 = crop(x, self.img_H, self.img_W)

            x = tf.concat([block3, up3], axis=3)
            x = self.sk3_conv1(x)
            x = self.sk3_bn1(x)
            x = self.sk3_convtran1(x)
            deconv3 = crop(x, self.img_H, self.img_W)

            x = tf.concat([block4, up2], axis=3)
            x = self.sk4_conv1(x)
            x = self.sk4_bn1(x)
            x = self.sk4_convtran1(x)
            deconv4 = crop(x, self.img_H, self.img_W)

            x = tf.concat([block5, up1], axis=3)
            x = self.sk5_conv1(x)
            x = self.sk5_bn1(x)
            x = self.sk5_convtran1(x)
            deconv5 = crop(x, self.img_H, self.img_W)

            mergeall = tf.concat([deconv1, deconv2, deconv3, deconv4, deconv5], axis=-1)

            output = self.conv10(mergeall)
            return output, c
        else:
            if c.numpy()[0][0]>0.5:
                x = self.up1([pool5, mask5])
                x = self.up1_conv1(x)
                x = self.up1_bn1(x)
                x = self.up1_relu1(x)
                x = self.up1_conv2(x)
                x = self.up1_bn2(x)
                x = self.up1_relu2(x)
                x = self.up1_conv3(x)
                x = self.up1_bn3(x)
                up1 = self.up1_relu3(x)

                x = self.up2([up1, mask4])
                x = self.up2_conv1(x)
                x = self.up2_bn1(x)
                x = self.up2_relu1(x)
                x = self.up2_conv2(x)
                x = self.up2_bn2(x)
                x = self.up2_relu2(x)
                x = self.up2_conv3(x)
                x = self.up2_bn3(x)
                up2 = self.up2_relu3(x)

                x = self.up3([up2, mask3])
                x = self.up3_conv1(x)
                x = self.up3_bn1(x)
                x = self.up3_relu1(x)
                x = self.up3_conv2(x)
                x = self.up3_bn2(x)
                x = self.up3_relu2(x)
                x = self.up3_conv3(x)
                x = self.up3_bn3(x)
                up3 = self.up3_relu3(x)

                x = self.up4([up3, mask2])
                x = self.up4_conv1(x)
                x = self.up4_bn1(x)
                x = self.up4_relu1(x)
                x = self.up4_conv2(x)
                x = self.up4_bn2(x)
                up4 = self.up4_relu2(x)

                x = self.up5([up4, mask1])
                x = self.up5_conv1(x)
                x = self.up5_bn1(x)
                x = self.up5_relu1(x)
                x = self.up5_conv2(x)
                x = self.up5_bn2(x)
                up5 = self.up5_relu2(x)

                x = tf.concat([block1, up5], axis=3)
                x = self.sk1_conv1(x)
                x = self.sk1_bn1(x)
                deconv1 = crop(x, self.img_H, self.img_W)

                x = tf.concat([block2, up4], axis=3)
                x = self.sk2_conv1(x)
                x = self.sk2_bn1(x)
                x = self.sk2_convtran1(x)
                deconv2 = crop(x, self.img_H, self.img_W)

                x = tf.concat([block3, up3], axis=3)
                x = self.sk3_conv1(x)
                x = self.sk3_bn1(x)
                x = self.sk3_convtran1(x)
                deconv3 = crop(x, self.img_H, self.img_W)

                x = tf.concat([block4, up2], axis=3)
                x = self.sk4_conv1(x)
                x = self.sk4_bn1(x)
                x = self.sk4_convtran1(x)
                deconv4 = crop(x, self.img_H, self.img_W)

                x = tf.concat([block5, up1], axis=3)
                x = self.sk5_conv1(x)
                x = self.sk5_bn1(x)
                x = self.sk5_convtran1(x)
                deconv5 = crop(x, self.img_H, self.img_W)

                mergeall = tf.concat([deconv1, deconv2, deconv3, deconv4, deconv5], axis=-1)

                output = self.conv10(mergeall)
                return output,c
            else:
                return self.black,c

if __name__=='__main__':
    img=np.random.uniform(low=0, high=1.0, size=(1,160,160,1))
    img=img.astype(np.float32)
    model = Deepcrack(input_shape=(1,160,160,1))
    model.summary()
    #model.load_weights("../weights_back/deepcrack_Crack206_160&160_11-18-7_29c.h5",by_name=True)
    aa=model.predict(img)
    print(aa)
    aa=model(img)
    print(aa)


