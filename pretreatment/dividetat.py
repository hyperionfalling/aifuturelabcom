import shutil  


root_path = 'D:/360Downloads/dataset/image_scene_training/'
txtfile = root_path + 'label.txt'
fr = open(txtfile)
j=0
for i in fr.readlines():
    item = i.split(' ')
    #print("D:/360Downloads/dataset/image_scene_training/data/"+ item[0])

    if int(j) % 5 == 0:
        shutil.copy("D:/360Downloads/dataset/image_scene_training/data/"+ item[0],"D:/360Downloads/dataset/image_scene_training/testdata")
    else :
        shutil.copy("D:/360Downloads/dataset/image_scene_training/data/"+ item[0],"D:/360Downloads/dataset/image_scene_training/traindata")
    j=j+1

    