import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio
import datetime
from PIL import Image 

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

root_path = 'D:/360Downloads/dataset/image_scene_training/'
tfrecords_filename = root_path + 'tfrecords/train.tfrecords'
print(tfrecords_filename)

height = 300
width = 300
txtfile = root_path + 'label.txt'
sess = tf.Session

j = 0
nf = 0
nct = 0
starttime = datetime.datetime.now()
'''
fr = open(txtfile)

for i in fr.readlines():
    item = i.split(' ')
    if j % 5 != 0 :
        if nct == 0 :
            writer = tf.python_io.TFRecordWriter("tfreords/"+"train"+str(nf)+".tfrecords")
        img = np.float64(misc.imread(root_path +'traindata/'+ item[0]))
        resized = img.resize((500,500,3))
        shape = np.array(img.shape, np.int32)
        #print(shape)
        label = int(item[1])
        img_raw = img.tostring()
        nct = nct + 1
    
     
        example = tf.train.Example(features=tf.train.Features(feature={
            'h': _int64_feature(shape[0]),
            'w': _int64_feature(shape[1]),
            'c': _int64_feature(shape[2]),
            'image_raw': _bytes_feature(img_raw),
            'label': _int64_feature(label)}))
    
        writer.write(example.SerializeToString())
    j = j + 1
    if nct == 1662 :
        nct = 0
        nf = nf + 1
        writer.close()


fr.close()
#writer.close()
'''
j = 0
fr = open(txtfile)
writer = tf.python_io.TFRecordWriter("tfreords/"+"test.tfrecords")
for i in fr.readlines():
    item = i.split(' ')
    if j % 5 == 0 :
        #writer = tf.python_io.TFRecordWriter("tfreords/"+"test.tfrecords")
        img = np.float64(misc.imread(root_path +'testdata/'+ item[0]))
        resized = img.resize((500,500,3))
        
        label = int(item[1])
        img_raw = img.tostring()
        shape = np.array(img.shape, np.int32)
        #print(shape)
        example = tf.train.Example(features=tf.train.Features(feature={
            'h': _int64_feature(shape[0]),
            'w': _int64_feature(shape[1]),
            'c': _int64_feature(shape[2]),
            'image_raw': _bytes_feature(img_raw),
            'label': _int64_feature(label)}))
    
        writer.write(example.SerializeToString())
    j = j + 1


writer.close()

fr.close()

endtime = datetime.datetime.now()
print (endtime - starttime)