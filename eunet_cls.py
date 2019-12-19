import tensorflow as tf
from tensorflow import keras
import datetime as dt
import numpy as np
import tensorflow as tf
import time
tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


class _Conv2d_block(tf.keras.Model):
    def __init__(self, n_filters, kernel_size,name):
        super(_Conv2d_block, self).__init__(name='')
        self.conv1=keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',name=name+"_conv1",padding='same')
        self.bn1  =keras.layers.BatchNormalization()
        self.relu1=keras.layers.Activation('relu')
        self.conv2=keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",name=name+"_conv2",padding="same")
        self.bn2  =keras.layers.BatchNormalization()
        self.relu2=keras.layers.Activation("relu")

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Unet_org_test(tf.keras.Model):
    def __init__(self,input_shape):
        super(Unet_org, self).__init__()

        def Conv2d_block( n_filters, kernel_size,name):
            return _Conv2d_block( n_filters, kernel_size,name)

        self.block1 = Conv2d_block(64, 3, name="block1")
        self.pool1 = keras.layers.MaxPooling2D((2, 2), name='block1_pool')

        self.block2 = Conv2d_block(128, 3, name="block2")
        self.pool2 = keras.layers.MaxPooling2D((2, 2), name='block2_pool')

        self.block3 = Conv2d_block(256, 3, name="block3")
        self.pool3 = keras.layers.MaxPooling2D((2, 2), name='block3_pool')

        self.block4 = Conv2d_block(512, 3, name="block4")
        self.pool4 = keras.layers.MaxPooling2D((2, 2), name='block4_pool')

        self.block5 = Conv2d_block(1024, 3, name="block5")

        self.up1 =keras.layers.UpSampling2D(size=(2, 2))
        #self.merge1 = keras.layers.concatenate(axis=3)
        self.block6 = Conv2d_block(512, 3, name='blockup1')

        self.up2 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge2 = keras.layers.concatenate(axis = 3)
        self.block7 = Conv2d_block(256, 3, name='blockup2')

        self.up3 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge3 = keras.layers.concatenate(axis = 3)
        self.block8 = Conv2d_block(128, 3, name='blockup3')

        self.up4 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge4 = keras.layers.concatenate(axis = 3)
        self.block9 = Conv2d_block(64, 3, name='blockup4')

        self.conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')
        self.build(input_shape)

    def call(self, inputs):
        #print(tf.executing_eagerly())
        block1 = self.block1(inputs)
        pool1 = self.pool1(block1)
        block2 = self.block2(pool1)
        pool2 = self.pool2(block2)
        block3 = self.block3(pool2)
        pool3 = self.pool3(block3)
        block4 = self.block4(pool3)
        pool4 = self.pool4(block4)

        block5 = self.block5(pool4)

        up1=self.up1(block5)
        merge1=tf.concat([block4,up1],axis=3)
        block6 = self.block6(merge1)

        up2=self.up2(block6)
        merge2=tf.concat([block3,up2],axis=3)
        block7 = self.block7(merge2)

        up3=self.up3(block7)
        merge3=tf.concat([block2,up3],axis=3)
        block8 = self.block8(merge3)

        up4=self.up4(block8)
        merge4=tf.concat([block1,up4],axis=3)
        block9 = self.block9(merge4)

        conv10=self.conv10(block9)
        return conv10

class Unet_org(tf.keras.Model):
    def __init__(self,input_shape):
        super(Unet_org, self).__init__()
        print(tf.executing_eagerly())
        self.block1_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="block1"+"_conv1",padding='same')
        self.block1_bn1  =keras.layers.BatchNormalization(name="block1"+"_bn1")
        self.block1_relu1=keras.layers.Activation('relu')
        self.block1_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="block1"+"_conv2",padding="same")
        self.block1_bn2  =keras.layers.BatchNormalization(name="block1"+"_bn2")
        self.block1_relu2=keras.layers.Activation("relu")
        self.pool1 = keras.layers.MaxPooling2D((2, 2), name='block1_pool')

        self.block2_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="block2"+"_conv1",padding='same')
        self.block2_bn1  =keras.layers.BatchNormalization(name="block2"+"_bn1")
        self.block2_relu1=keras.layers.Activation('relu')
        self.block2_conv2=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="block2"+"_conv2",padding="same")
        self.block2_bn2  =keras.layers.BatchNormalization(name="block2"+"_bn2")
        self.block2_relu2=keras.layers.Activation("relu")
        self.pool2 = keras.layers.MaxPooling2D((2, 2), name='block2_pool')

        self.block3_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="block3"+"_conv1",padding='same')
        self.block3_bn1  =keras.layers.BatchNormalization(name="block3"+"_bn1")
        self.block3_relu1=keras.layers.Activation('relu')
        self.block3_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="block3"+"_conv2",padding="same")
        self.block3_bn2  =keras.layers.BatchNormalization(name="block3"+"_bn2")
        self.block3_relu2=keras.layers.Activation("relu")
        self.pool3 = keras.layers.MaxPooling2D((2, 2), name='block3_pool')

        self.block4_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="block4"+"_conv1",padding='same')
        self.block4_bn1  =keras.layers.BatchNormalization(name="block4"+"_bn1")
        self.block4_relu1=keras.layers.Activation('relu')
        self.block4_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block4"+"_conv2",padding="same")
        self.block4_bn2  =keras.layers.BatchNormalization(name="block4"+"_bn2")
        self.block4_relu2=keras.layers.Activation("relu")
        self.pool4 = keras.layers.MaxPooling2D((2, 2), name='block4_pool')

        self.block5_conv1=keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), kernel_initializer='he_normal',name="block5"+"_conv1",padding='same')
        self.block5_bn1  =keras.layers.BatchNormalization(name="block5"+"_bn1")
        self.block5_relu1=keras.layers.Activation('relu')
        self.block5_conv2=keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), kernel_initializer="he_normal",name="block5"+"_conv2",padding="same")
        self.block5_bn2  =keras.layers.BatchNormalization(name="block5"+"_bn2")
        self.block5_relu2=keras.layers.Activation("relu")

        self.up1 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge1 = keras.layers.concatenate(axis=3)
        self.block6_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup1"+"_conv1",padding='same')
        self.block6_bn1  =keras.layers.BatchNormalization(name="blockup1"+"_bn1")
        self.block6_relu1=keras.layers.Activation('relu')
        self.block6_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup1"+"_conv2",padding="same")
        self.block6_bn2  =keras.layers.BatchNormalization(name="blockup1"+"_bn2")
        self.block6_relu2=keras.layers.Activation("relu")

        self.up2 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge2 = keras.layers.concatenate(axis = 3)
        self.block7_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup2"+"_conv1",padding='same')
        self.block7_bn1  =keras.layers.BatchNormalization(name="blockup2"+"_bn1")
        self.block7_relu1=keras.layers.Activation('relu')
        self.block7_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup2"+"_conv2",padding="same")
        self.block7_bn2  =keras.layers.BatchNormalization(name="blockup2"+"_bn2")
        self.block7_relu2=keras.layers.Activation("relu")

        self.up3 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge3 = keras.layers.concatenate(axis = 3)
        self.block8_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup3"+"_conv1",padding='same')
        self.block8_bn1  =keras.layers.BatchNormalization(name="blockup3"+"_bn1")
        self.block8_relu1=keras.layers.Activation('relu')
        self.block8_conv2=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup3"+"_conv2",padding="same")
        self.block8_bn2  =keras.layers.BatchNormalization(name="blockup3"+"_bn2")
        self.block8_relu2=keras.layers.Activation("relu")

        self.up4 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge4 = keras.layers.concatenate(axis = 3)
        self.block9_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup4"+"_conv1",padding='same')
        self.block9_bn1  =keras.layers.BatchNormalization(name="blockup4"+"_bn1")
        self.block9_relu1=keras.layers.Activation('relu')
        self.block9_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup4"+"_conv2",padding="same")
        self.block9_bn2  =keras.layers.BatchNormalization(name="blockup4"+"_bn2")
        self.block9_relu2=keras.layers.Activation("relu")

        self.conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid',name='sfconv1')

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

        pool1 = self.pool1(block1)
        x = self.block2_conv1(pool1)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        block2 = self.block2_relu2(x)

        pool2 = self.pool2(block2)
        x = self.block3_conv1(pool2)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        block3 = self.block3_relu2(x)

        pool3 = self.pool3(block3)
        x = self.block4_conv1(pool3)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        block4 = self.block4_relu2(x)
        pool4 = self.pool4(block4)

        x = self.block5_conv1(pool4)
        x = self.block5_bn1(x)
        x = self.block5_relu1(x)
        x = self.block5_conv2(x)
        x = self.block5_bn2(x)
        block5 = self.block5_relu2(x)

        up1 = self.up1(block5)
        merge1 = tf.concat([block4, up1], axis=3)
        x = self.block6_conv1(merge1)
        x = self.block6_bn1(x)
        x = self.block6_relu1(x)
        x = self.block6_conv2(x)
        x = self.block6_bn2(x)
        block6 = self.block6_relu2(x)

        up2 = self.up2(block6)
        merge2 = tf.concat([block3, up2], axis=3)
        x = self.block7_conv1(merge2)
        x = self.block7_bn1(x)
        x = self.block7_relu1(x)
        x = self.block7_conv2(x)
        x = self.block7_bn2(x)
        block7 = self.block7_relu2(x)

        up3 = self.up3(block7)
        merge3 = tf.concat([block2, up3], axis=3)
        x = self.block8_conv1(merge3)
        x = self.block8_bn1(x)
        x = self.block8_relu1(x)
        x = self.block8_conv2(x)
        x = self.block8_bn2(x)
        block8 = self.block8_relu2(x)

        up4 = self.up4(block8)
        merge4 = tf.concat([block1, up4], axis=3)
        x = self.block9_conv1(merge4)
        x = self.block9_bn1(x)
        x = self.block9_relu1(x)
        x = self.block9_conv2(x)
        x = self.block9_bn2(x)
        block9 = self.block9_relu2(x)

        conv10 = self.conv10(block9)
        return conv10



class Unet_org_cls(tf.keras.Model):
    def __init__(self,input_shape):
        super(Unet_org_cls, self).__init__()
        print(tf.executing_eagerly())
        self.block1_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="block1"+"_conv1",padding='same')
        self.block1_bn1  =keras.layers.BatchNormalization(name="block1"+"_bn1")
        self.block1_relu1=keras.layers.Activation('relu')
        self.block1_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="block1"+"_conv2",padding="same")
        self.block1_bn2  =keras.layers.BatchNormalization(name="block1"+"_bn2")
        self.block1_relu2=keras.layers.Activation("relu")        
        self.pool1 = keras.layers.MaxPooling2D((2, 2), name='block1_pool')

        self.block2_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="block2"+"_conv1",padding='same')
        self.block2_bn1  =keras.layers.BatchNormalization(name="block2"+"_bn1")
        self.block2_relu1=keras.layers.Activation('relu')
        self.block2_conv2=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="block2"+"_conv2",padding="same")
        self.block2_bn2  =keras.layers.BatchNormalization(name="block2"+"_bn2")
        self.block2_relu2=keras.layers.Activation("relu")
        self.pool2 = keras.layers.MaxPooling2D((2, 2), name='block2_pool')

        self.block3_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="block3"+"_conv1",padding='same')
        self.block3_bn1  =keras.layers.BatchNormalization(name="block3"+"_bn1")
        self.block3_relu1=keras.layers.Activation('relu')
        self.block3_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="block3"+"_conv2",padding="same")
        self.block3_bn2  =keras.layers.BatchNormalization(name="block3"+"_bn2")
        self.block3_relu2=keras.layers.Activation("relu")
        self.pool3 = keras.layers.MaxPooling2D((2, 2), name='block3_pool')

        self.block4_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="block4"+"_conv1",padding='same')
        self.block4_bn1  =keras.layers.BatchNormalization(name="block4"+"_bn1")
        self.block4_relu1=keras.layers.Activation('relu')
        self.block4_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="block4"+"_conv2",padding="same")
        self.block4_bn2  =keras.layers.BatchNormalization(name="block4"+"_bn2")
        self.block4_relu2=keras.layers.Activation("relu")
        self.pool4 = keras.layers.MaxPooling2D((2, 2), name='block4_pool')

        self.block5_conv1=keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), kernel_initializer='he_normal',name="block5"+"_conv1",padding='same')
        self.block5_bn1  =keras.layers.BatchNormalization(name="block5"+"_bn1")
        self.block5_relu1=keras.layers.Activation('relu')
        self.block5_conv2=keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), kernel_initializer="he_normal",name="block5"+"_conv2",padding="same")
        self.block5_bn2  =keras.layers.BatchNormalization(name="block5"+"_bn2")
        self.block5_relu2=keras.layers.Activation("relu")

        self.up1 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge1 = keras.layers.concatenate(axis=3)
        self.block6_conv1=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup1"+"_conv1",padding='same')
        self.block6_bn1  =keras.layers.BatchNormalization(name="blockup1"+"_bn1")
        self.block6_relu1=keras.layers.Activation('relu')
        self.block6_conv2=keras.layers.Conv2D(filters=512, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup1"+"_conv2",padding="same")
        self.block6_bn2  =keras.layers.BatchNormalization(name="blockup1"+"_bn2")
        self.block6_relu2=keras.layers.Activation("relu")

        self.up2 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge2 = keras.layers.concatenate(axis = 3)
        self.block7_conv1=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup2"+"_conv1",padding='same')
        self.block7_bn1  =keras.layers.BatchNormalization(name="blockup2"+"_bn1")
        self.block7_relu1=keras.layers.Activation('relu')
        self.block7_conv2=keras.layers.Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup2"+"_conv2",padding="same")
        self.block7_bn2  =keras.layers.BatchNormalization(name="blockup2"+"_bn2")
        self.block7_relu2=keras.layers.Activation("relu")

        self.up3 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge3 = keras.layers.concatenate(axis = 3)
        self.block8_conv1=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup3"+"_conv1",padding='same')
        self.block8_bn1  =keras.layers.BatchNormalization(name="blockup3"+"_bn1")
        self.block8_relu1=keras.layers.Activation('relu')
        self.block8_conv2=keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup3"+"_conv2",padding="same")
        self.block8_bn2  =keras.layers.BatchNormalization(name="blockup3"+"_bn2")
        self.block8_relu2=keras.layers.Activation("relu")

        self.up4 = keras.layers.UpSampling2D(size = (2,2))
        #self.merge4 = keras.layers.concatenate(axis = 3)
        self.block9_conv1=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',name="blockup4"+"_conv1",padding='same')
        self.block9_bn1  =keras.layers.BatchNormalization(name="blockup4"+"_bn1")
        self.block9_relu1=keras.layers.Activation('relu')
        self.block9_conv2=keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer="he_normal",name="blockup4"+"_conv2",padding="same")
        self.block9_bn2  =keras.layers.BatchNormalization(name="blockup4"+"_bn2")
        self.block9_relu2=keras.layers.Activation("relu")

        self.conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid',name='sfconv1')

        self.cls_max1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool3')
        self.cls_conv1 = keras.layers.Conv2D(256, (3, 3), padding='same', name='block5_conv3')
        self.cls_bn1  = keras.layers.BatchNormalization(name='block5_bn3')
        self.cls_relu1 = keras.layers.Activation('relu')
        self.cls_conv2 = keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same',name='sfconv2')
        self.cls_gmax2 = keras.layers.GlobalMaxPooling2D(name='gmax_pool1')

        #self.cls_conv1 = keras.layers.Conv2D(512, (3, 3), padding='same', name='block5_conv3')
        #self.cls_bn1  = keras.layers.BatchNormalization(name='block5_bn3')
        #self.cls_relu1 = keras.layers.Activation('relu')
        #self.cls_max1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
        #self.cls_conv2 = keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same',name='sfconv2')
        #self.cls_gmax2 = keras.layers.GlobalMaxPooling2D()

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

        pool1 = self.pool1(block1)
        x = self.block2_conv1(pool1)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        block2 = self.block2_relu2(x)

        pool2 = self.pool2(block2)
        x = self.block3_conv1(pool2)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        block3 = self.block3_relu2(x)

        pool3 = self.pool3(block3)
        x = self.block4_conv1(pool3)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        block4 = self.block4_relu2(x)
        pool4 = self.pool4(block4)

        x = self.block5_conv1(pool4)
        x = self.block5_bn1(x)
        x = self.block5_relu1(x)
        x = self.block5_conv2(x)
        x = self.block5_bn2(x)
        block5 = self.block5_relu2(x)

        #c=self.cls_conv1(block5)
        #c=self.cls_bn1(c)
        #c=self.cls_relu1(c)
        #c=self.cls_max1(c)

        c=self.cls_max1(block5)
        c=self.cls_conv1(c)
        c=self.cls_bn1(c)
        c=self.cls_relu1(c)

        c=self.cls_conv2(c)
        c=self.cls_gmax2(c)

        if tf.executing_eagerly()==False:
            print("not executing_eagerly")#
            up1 = self.up1(block5)
            merge1 = tf.concat([block4, up1], axis=3)
            x = self.block6_conv1(merge1)
            x = self.block6_bn1(x)
            x = self.block6_relu1(x)
            x = self.block6_conv2(x)
            x = self.block6_bn2(x)
            block6 = self.block6_relu2(x)

            up2 = self.up2(block6)
            merge2 = tf.concat([block3, up2], axis=3)
            x = self.block7_conv1(merge2)
            x = self.block7_bn1(x)
            x = self.block7_relu1(x)
            x = self.block7_conv2(x)
            x = self.block7_bn2(x)
            block7 = self.block7_relu2(x)

            up3 = self.up3(block7)
            merge3 = tf.concat([block2, up3], axis=3)
            x = self.block8_conv1(merge3)
            x = self.block8_bn1(x)
            x = self.block8_relu1(x)
            x = self.block8_conv2(x)
            x = self.block8_bn2(x)
            block8 = self.block8_relu2(x)

            up4 = self.up4(block8)
            merge4 = tf.concat([block1, up4], axis=3)
            x = self.block9_conv1(merge4)
            x = self.block9_bn1(x)
            x = self.block9_relu1(x)
            x = self.block9_conv2(x)
            x = self.block9_bn2(x)
            block9 = self.block9_relu2(x)

            conv10 = self.conv10(block9)
            return conv10, c
        else:
            if c.numpy()[0][0]>0.5:
                up1=self.up1(block5)
                merge1=tf.concat([block4,up1],axis=3)
                x = self.block6_conv1(merge1)
                x = self.block6_bn1(x)
                x = self.block6_relu1(x)
                x = self.block6_conv2(x)
                x = self.block6_bn2(x)
                block6 = self.block6_relu2(x)

                up2=self.up2(block6)
                merge2=tf.concat([block3,up2],axis=3)
                x = self.block7_conv1(merge2)
                x = self.block7_bn1(x)
                x = self.block7_relu1(x)
                x = self.block7_conv2(x)
                x = self.block7_bn2(x)
                block7 = self.block7_relu2(x)

                up3=self.up3(block7)
                merge3=tf.concat([block2,up3],axis=3)
                x = self.block8_conv1(merge3)
                x = self.block8_bn1(x)
                x = self.block8_relu1(x)
                x = self.block8_conv2(x)
                x = self.block8_bn2(x)
                block8 = self.block8_relu2(x)

                up4=self.up4(block8)
                merge4=tf.concat([block1,up4],axis=3)
                x = self.block9_conv1(merge4)
                x = self.block9_bn1(x)
                x = self.block9_relu1(x)
                x = self.block9_conv2(x)
                x = self.block9_bn2(x)
                block9 = self.block9_relu2(x)

                conv10=self.conv10(block9)
                return conv10,c
            else:
                return self.black,c


if __name__ == '__main__':
    img=np.random.uniform(low=0, high=1.0, size=(1,160,160,1))
    img=img.astype(np.float32)
    model = Unet_org_cls(input_shape=(1,160,160,1))
    model.summary()
    #model.load_weights("../weights_back/unet_adbn_cls_Crack206_160&160_8-31-20_19.h5",by_name=True)
    aa,c=model.predict(img)
    start_time = time.time()
    aa,c=model.predict(img)
    end_time = time.time()
    run_time = (end_time - start_time) * 1000
    print("static_model runtime=%.4fMS,FPS=%d" % (run_time, 1000 / run_time))
    #print(aa,c)

    start_time = time.time()
    aa,c=model(img)
    end_time = time.time()
    run_time = (end_time - start_time) * 1000
    print("eager_model runtime=%.4fMS,FPS=%d" % (run_time, 1000 / run_time))
    #print(aa,c)


