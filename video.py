import cv2
import os


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
####################################################################################################        
    if c > -1 and c != prev_char:
        cur_char =c
    if(prev_char != c):
        prev_char = cur_char
    
    if(c == ord('0')):
        while(os.path.exists("./DATA/Video/0/0train_{}.mp4".format(i))):
            i += 1

        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/0/0train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False
    elif(c == ord('1')):
        while(os.path.exists("./DATA/Video/1/1train_{}.mp4".format(i))):
            i += 1
        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/1/1train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False
    elif(c == ord('2')):
        while(os.path.exists("./DATA/Video/2/2train_{}.mp4".format(i))):
            i += 1
        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/2/2train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False    
    elif(c == ord('3')):
        while(os.path.exists("./DATA/Video/3/3train_{}.mp4".format(i))):
            i += 1
        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/3/3train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False
    elif(c == ord('4')):
        while(os.path.exists("./DATA/Video/4/4train_{}.mp4".format(i))):
            i += 1

        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/4/4train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False
    elif(c == ord('5')):
        while(os.path.exists("./DATA/Video/5/5train_{}.mp4".format(i))):
            i += 1

        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/5/5train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False    
    elif(c == ord('6')):
        while(os.path.exists("./DATA/Video/6/6train_{}.mp4".format(i))):
            i += 1

        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/6/6train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False        
    elif(c == ord('7')):
        while(os.path.exists("./DATA/Video/7/7train_{}.mp4".format(i))):
            i += 1

        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/7/7train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False        
    elif(c == ord('8')):
        while(os.path.exists("./DATA/Video/8/8train_{}.mp4".format(i))):
            i += 1

        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/8/8train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False        
    elif(c == ord('9')):
        while(os.path.exists("./DATA/Video/9/9train_{}.mp4".format(i))):
            i += 1

        if(not Record): 
            video = cv2.VideoWriter("DATA/Video/9/9train_{}.mp4".format(i), fourcc, 30.0, (image.shape[1], image.shape[0]))
            Record = True
            i=0
        else:
            video.release()
            i =0
            Record = False            
    else:
        print('Not')


        #print(prev_char)

    if(Record):
        video.write(image)

    if c == 27: break