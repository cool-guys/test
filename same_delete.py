import os, sys
import hashlib

file_list = []
cnt = 0
path = '../Video/train_set'

def filelist(path):
    global file_list
    global cnt
    file_list = os.listdir(path)
    file_list.sort()
    cnt = len(file_list)
def getHash(path, blocksize=65536):
    try:
        file = open(path, 'rb')
    except os.error:
        print('file is deleted')
        return 'x'
    hasher = hashlib.md5()
    buf = file.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf =  file.read(blocksize)
    file.close()
    return hasher.hexdigest()
for i in range(26):
    path = '../Video/train_set/{}'.format(chr(97+i))
    filelist(path)
    for i in file_list:
        DPath = path + '/' + i
        if not DPath == path + '/' + 'backup':
            continue
        n1 = os.path.getsize(DPath)
        hash1 = getHash(DPath)
        for j in file_list:
            if i==j: continue
            D2Path = path + '/' + j
            if not D2Path == path + '/' + 'backup':
                continue
            try:
                n2 = os.path.getsize(D2Path)
            except os.error:
                print('file is deleted')
                continue
            if n1==n2:
                hash2 = getHash(D2Path)
                if hash1 == hash2:
                    os.remove(D2Path)