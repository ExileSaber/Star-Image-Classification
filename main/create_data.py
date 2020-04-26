import os
import numpy as np

names = ['anqi', 'wangchengxuan', 'xujiaqi', 'yushuxin', 'zhaoxiaotang']
path = '..\\data'
save_dir = '..\\dataset'
datas = []
for root, dirs, files in os.walk(path):
    if files != None:
        for thefile in files:
            for i in range(len(names)):
                if names[i] in root:
                    data = root + '\\' + thefile + ' ' + str(i)
                    datas.append(data)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
datas = np.array(datas)
print(datas)

np.random.shuffle(datas)
train_list = datas[:int(0.8*len(datas))]
val_list = datas[int(0.8*len(datas)): -5]
test_list = datas[-5:]
label_list = np.array(names)

with open("..\\dataset\\test_list.txt", 'ab') as abc:  # 写入numpy.ndarray数据
    np.savetxt(abc, test_list, fmt='%s')

with open("..\\dataset\\val_list.txt", 'ab') as abc:  # 写入numpy.ndarray数据
    np.savetxt(abc, val_list, fmt='%s')

with open("..\\dataset\\train_list.txt", 'ab') as abc:  # 写入numpy.ndarray数据
    np.savetxt(abc, train_list, fmt='%s')

with open("..\\dataset\\label_list.txt", 'ab') as abc:  # 写入numpy.ndarray数据
    np.savetxt(abc, label_list, fmt='%s')
