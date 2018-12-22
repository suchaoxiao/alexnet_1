import keras
import argparse
import numpy as np
from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Dense,Input,add,Activation,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler,TensorBoard,ModelCheckpoint
from keras.models import Model
from keras import optimizers,regularizers
from keras import backend as K

if('tensorflow'==K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess=tf.Session(config=config)

parser=argparse.ArgumentParser()
parser.add_argument('-b','--batch_size',type=int,default=128,metavar='NUMBER',
                    help='batch size(default:128)')
parser.add_argument('-e','--epochs',type=int,default=200,metavar='NUMBER',
                    help='epochs(default:200)')
parser.add_argument('-n','--stack_n',type=int,default=5,metavar='NUMBER',
                    help='stack number n,total layers=6*n+2(default n =5)')
parser.add_argument('-d','--dataset',type=str,default='cifar10',metavar='STRING',
                    help='dataset,(default:cifar10)')
args=parser.parse_args()

stack_n=args.stack_n
layers=6*stack_n+2
num_classes=10
img_rows,img_cols=32,32
img_channels=3
batch_size=args.batch_size
epochs=args.epochs
iterations=50000//batch_size+1
weight_decay=1e-4

def color_preprocessing(x_train,x_test):
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    mean=[125.307,122.95,113.865]
    std=[62.9932,62.0887,66.7048]
    for i in range(3):
        x_train[:,:,:,i]=(x_train[:,:,:,i]-mean[i])/std[i]
        x_test[:,:,:,i]=(x_test[:,:,:,i]-mean[i])/std[i]
    return x_train,x_test

def scheduler(epoch):
    if epoch<81:
        return 0.1
    elif epoch<122:
        return 0.01
    else:
        return 0.001

def residual_network(img_input,classes_num=10,stack_n=5):
    def residual_block(x,o_filters,increase=False):
        stride=(1,1)
        if increase:
            stride=(2,2)

        o1=Activation('relu')(BatchNormalization(momentum=0.9,epsilon=1e-5)(x))
        conv_1=Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2=Activation('relu')(BatchNormalization(momentum=0.9,epsilon=1e-5)(conv_1))
        conv_2=Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay))(o2)

        if increase:
            projection=Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block=add([conv_2,projection])
        else:
            block=add([conv_2,x])
        return block
    x=Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
             kernel_initializer='he_normal',
             kernel_regularizer=regularizers.l2(weight_decay))(img_input)
    #input:32x32x16  output:32x32x16
    for _ in range(stack_n):
        x=residual_block(x,16,False)
    #input:32x32x16 output:16x16x32
    x=residual_block(x,32,True)
    for _ in range(1,stack_n):
        x=residual_block(x,32,False)
    #input:16x16x32  output:8x8x64
    x=residual_block(x,64,True)
    for _ in range(1,stack_n):
        x=residual_block(x,64,False)
    x=BatchNormalization(momentum=0.9,epsilon=1e-5)(x)
    x=Activation('relu')(x)
    x=GlobalAveragePooling2D()(x)

    #input:64,output:10
    x=Dense(classes_num,activation='softmax',kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

if __name__=='__main__':
    print('==========================')
    print('model:residual network({:2d} layers)'.format(6*stack_n+2))
    print('Batch size({:3d})'.format(batch_size))
    print('weight decay:{:.4f}'.format(weight_decay))
    print('epochs{:3d}'.format(epochs))
    print('dataset{:}'.format(args.dataset))

    print('==load data...==')

    # global num_classes
    if args.dataset=='cifar100':
        num_classes=100
        (x_train,y_train),(x_test,y_test)=cifar100.load_data()
    else:
        (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    y_train=keras.utils.to_categorical(y_train,num_classes)
    y_test=keras.utils.to_categorical(y_test,num_classes)

    print('==done==\n==color preprocessing...==')
    x_train,x_test=color_preprocessing(x_train,x_test)
    print('==done==\n==build model...==')
    img_input=Input(shape=(img_rows,img_cols,img_channels))
    output=residual_network(img_input,num_classes,stack_n)
    resnet=Model(img_input,output)

    print(resnet.summary())

    sgd=optimizers.SGD(lr=.1,momentum=0.9,nesterov=True)
    resnet.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    cbks=[TensorBoard(log_dir='./resnet_{:d}_{}/'.format(layers,args.dataset),histogram_freq=0),
          LearningRateScheduler(scheduler)]

    #set data augmentation
    print('==using real-time data augmentation start train...==')
    datagen=ImageDataGenerator(horizontal_flip=True,
                               width_shift_range=0.125,
                               height_shift_range=0.125,
                               fill_mode='constant',cval=0.)
    datagen.fit(x_train)

    resnet.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test,y_test))
    resnet.save('resnet_{:d}_{}.h5'.format(layers,args.dataset))