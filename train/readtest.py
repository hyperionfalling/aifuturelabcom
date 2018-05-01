# -*- coding: utf-8 -*-
import tensorflow as tf
with tf.Session() as sess:  
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer(["D:/360Downloads/dataset/image_scene_training/tfrecords/test.tfrecords"],num_epochs=1)
    
    # 从文件中读出一个样例，也可以使用read_up_to函数一次性读取多个样例
    # _, serialized_example = reader.read(filename_queue)
    _, serialized_example = reader.read_up_to(filename_queue, 6) #读取6个样例
    # 解析读入的一个样例，如果需要解析多个样例，可以用parse_example函数
    # features = tf.parse_single_example(serialized_example, features={
    # 解析多个样例
    features = tf.parse_example(serialized_example, features={
        # TensorFlow提供两种不同的属性解析方法
        # 第一种是tf.FixedLenFeature,得到的解析结果为Tensor
        # 第二种是tf.VarLenFeature,得到的解析结果为SparseTensor，用于处理稀疏数据
        # 解析数据的格式需要与写入数据的格式一致
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'c': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),   #https://stackoverflow.com/questions/41921746/tensorflow-varlenfeature-vs-fixedlenfeature
        'name': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })
    
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)
    c = tf.cast(features['c'], tf.int32)
    name = features['name']
    image = tf.reshape(image, [500, 500, 3]) 
    label = tf.reshape(label, [1]) 
    #print(label.get_shape())
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, min_after_dequeue=10) 
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  
    sess.run(init_op)
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  
    coord.request_stop()  
    coord.join(threads)  
    sess.close()  
