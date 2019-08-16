import cv2
import os

if(not os.path.isdir("../DATA/Video")):
    os.mkdir("../DATA/Video")
else:
    pass

for i in range(26):
    if(not os.path.isdir("../DATA/Video/{}".format(chr(97+i)))):
        os.mkdir("../DATA/Video/{}".format(chr(97+i)))
    else:
        pass

cam = cv2.VideoCapture(0)
# Process and display images

c = cv2.waitKey(0)
data = []
i = 0
prev_char = -1 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
Record = False

while(True):
    ret_val, image = cam.read()


    cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", image)
    c = cv2.waitKey(1)

    if c > -1 and c != prev_char:
        cur_char =c
    if(prev_char != c):
        prev_char = cur_char
    
    for j in range(26):
        if(c == ord('a') + j):
            while(os.path.exists("./DATA/Video/{}/{}train_{}.mp4".format(chr(97+j),chr(97+j),i))):
                i += 1

            if(not Record): 
                video = cv2.VideoWriter("DATA/Video/{}/{}train_{}.mp4".format(chr(97+j),chr(97+j),i), fourcc, 30.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
            else:
                print("{}_train_{}.mp4 saved".format(chr(97+j),i))
                video.release()
                i =0
                Record = False
    
    if(Record):
        video.write(image)

    if c == 27: break