import numpy as np
import os
import shutil
#files = os.listdir('../Video/a')



def order_file():
    for i in range(26):
        counts = 0
        files = os.listdir('../Video/{}'.format(chr(97+i)))

        for name in files:
            if(name.endswith("mp4") and name[0] != '.' and name[7] != '_'):
                new_name = '../Video/{}/'.format(chr(97+i)) + "{}train_{}.mp4".format(chr(97+i),counts)
                new_name_2 = '../Video/{}/'.format(chr(97+i)) + "{}train_{}".format(chr(97+i),counts)
                name = '../Video/{}/'.format(chr(97+i)) + name
                name_2 = name[:-4]
                os.rename(name,new_name)
                os.rename(name_2,new_name_2)
                counts += 1

def name_change():
    counts = 0

    for i in range(26):
        files = os.listdir('../Video/{}'.format(chr(97+i)))
        for name in files:
            if(name.endswith("mp4") and name[7] != '_' and name[0] != '.'):
                new_name = '../Video/{}/'.format(chr(97+i)) + "{}train_{}.mp4".format(chr(97+i),counts + 15000)
                new_name_2 = '../Video/{}/'.format(chr(97+i)) + "{}train_{}".format(chr(97+i),counts + 15000)
                name = '../Video/{}/'.format(chr(97+i)) + name
                name_2 = name[:-4]
                os.rename(name,new_name)
                os.rename(name_2,new_name_2)
                counts += 1
def name_change_2():
    counts = 0 

    files = os.listdir('../Video/변경')
    for name in files:
        if(name.endswith("mp4") ):
            new_name = '../Video/변경/' + "otrain_{}.mp4".format(counts + 157)
            new_name_2 = '../Video/변경/' + "otrain_{}".format(counts + 157)
            name = '../Video/변경/' + name
            name_2 = name[:-4]
            os.rename(name,new_name)
            os.rename(name_2,new_name_2)
            counts += 1
def file_move():
    
    for i in range(26):
        l = 0
        counts = 0
        files = os.listdir('../Video/{}'.format(chr(97+i)))

        if(os.path.isdir('../Video/{}/{}'.format(chr(97+i),chr(97+i)))):
            pass
        else:
            os.mkdir('../Video/{}/{}'.format(chr(97+i),chr(97+i)))
        files_num = len(files)

        for name in files:
            if(name.endswith("mp4") and l < 20):
                pickle_name = '../Video/{}/'.format(chr(97+i)) + name[:-4]
                pickle_name_2 = '../Video/{}/{}/'.format(chr(97+i),chr(97+i)) + name[:-4]
                video_name = '../Video/{}/'.format(chr(97+i)) + name
                video_name_2 = '../Video/{}/{}/'.format(chr(97+i),chr(97+i)) + name

                shutil.move(pickle_name,pickle_name_2)
                shutil.move(video_name,video_name_2)
                l += 1
def name_change_3():
    for i in range(26):
        for j in range(500):
            name = "../Video/{}/test_{}".format(chr(97+i),j)
            name_2 = "../Video/{}/{}train_{}".format(chr(97+i),chr(97+i),j)

            os.rename(name,name_2)

name_change_3()
#order_file()
#file_move()