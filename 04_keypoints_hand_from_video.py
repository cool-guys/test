# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import pandas as pd
import numpy as np

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../../models/"
params["hand"] = True
#params["render_pose"] = 0
params["hand_render"] = 2
params["profile_speed"]= 10
# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
j = 0
datum = op.Datum()
# Read frames on directory
#imagePaths = op.get_images_on_directory(args[0].image_dir);
#start = time.time()
for i in range(10):
  while(os.path.exists("./DATA/Video/{}/{}train_{}.mp4".format(i,i,j))):
    j += 1
  for k in range(j):
    if(not(os.path.exists("./DATA/Video/{}/{}train_{}".format(i,i,k)))):
        cam = cv2.VideoCapture("./DATA/Video/{}/{}train_{}.mp4".format(i,i,k))
        data = []

        while(True):
            ret_val, image = cam.read()
            if(ret_val):
                datum.cvInputData = image

                opWrapper.emplaceAndPop([datum])
                handkeypoints = np.array(datum.handKeypoints)

                if(handkeypoints.shape !=(2,)):
                    handkeypoints = np.array([handkeypoints[1][0][0][0],handkeypoints[1][0][0][1]])

                #cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)

                data.append(handkeypoints.astype(int))
                dataframe = pd.DataFrame(data, columns= ['x','y'])
                dataframe['label'] = i
            else:
                break

        dataframe.to_csv("./DATA/Video/{}/{}train_{}".format(i,i,k), index=False)
        print("./DATA/Video/{}/{}train_{} saved".format(i,i,k))



'''
while(True):
    ret_val, image = cam.read()
    datum.cvInputData = image

    opWrapper.emplaceAndPop([datum])
    handkeypoints = np.array(datum.handKeypoints)
    #print(handkeypoints.shape)
    if(handkeypoints.shape !=(2,)):
        handkeypoints = np.array([handkeypoints[1][0][0][0],handkeypoints[1][0][0][1]])
    #else:
    #    print("warning",np.array(handkeypoints).shape)
    #    handkeypoints = np.array([0,0])

    #print(handkeypoints[1][0][0][0])
    if(datum.poseKeypoints.shape != () ):
        #print("Body keypoints: \n" + str(handkeypoints[0][0]))
        #data.append(handkeypoints.astype(int))
        print("OK")
    else:
        print("Not OK")

    

    if not args[0].no_display:
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
        c = cv2.waitKey(1)
####################################################################################################        
        if c > -1 and c != prev_char:
            cur_char =c
        if(prev_char != c):
            prev_char = cur_char

        if(c == ord('0')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 0
            while(os.path.exists("./DATA/0/0train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/0/0train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
        elif(c == ord('1')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 1
            while(os.path.exists("./DATA/1/1train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/1/1train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0

        elif(c == ord('2')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 2
            while(os.path.exists("./DATA/2/2train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/2/2train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
        
        elif(c == ord('3')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 3
            while(os.path.exists("./DATA/3/3train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/3/3train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
         
        elif(c == ord('4')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 4
            while(os.path.exists("./DATA/4/4train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/4/4train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
   
        elif(c == ord('5')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 5
            while(os.path.exists("./DATA/5/5train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/5/5train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
        
        elif(c == ord('6')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 6
            while(os.path.exists("./DATA/6/6train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/6/6train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
           
        elif(c == ord('7')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 7
            while(os.path.exists("./DATA/7/7train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/7/7train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
          
        elif(c == ord('8')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 8
            while(os.path.exists("./DATA/8/8train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/8/8train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
         
        elif(c == ord('9')):
            img = np.zeros((500, 500, 1), np.uint8)
            data.append(handkeypoints.astype(int))
            dataframe = pd.DataFrame(data, columns= ['x','y'])
            dataframe['label'] = 9
            while(os.path.exists("./DATA/9/9train_{}".format(i))):
                i += 1
            if(not Record): 
                video = cv2.VideoWriter("DATA/9/9train_{}.mp4".format(i), fourcc, 20.0, (image.shape[1], image.shape[0]))
                Record = True
                i=0
        else:
            if(prev_char == ord('0') and len(data) > 5  ):
                video.release()
                while(os.path.exists("./DATA/0/0train_{}".format(i))):
                    i += 1 
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]
                dataframe.to_csv("./DATA/0/0train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('1') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/1/1train_{}".format(i))):
                    i += 1 
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]    
                dataframe.to_csv("./DATA/1/1train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('2') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/2/2train_{}".format(i))):
                    i += 1
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]              
                dataframe.to_csv("./DATA/2/2train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('3') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/3/3train_{}".format(i))):
                    i += 1
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]
                dataframe.to_csv("./DATA/3/3train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('4') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/4/4train_{}".format(i))):
                    i += 1
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]                    
                dataframe.to_csv("./DATA/4/4train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)
            
                i=0
                Record = False
                data = []
            elif(prev_char == ord('5') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/5/5train_{}".format(i))):
                    i += 1
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]                     
                dataframe.to_csv("./DATA/5/5train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('6') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/6/6train_{}".format(i))):
                    i += 1 
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]                    
                dataframe.to_csv("./DATA/6/6train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('7') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/7/7train_{}".format(i))):
                    i += 1 
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]                    
                dataframe.to_csv("./DATA/7/7train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('8') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/8/8train_{}".format(i))):
                    i += 1 
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]                    
                dataframe.to_csv("./DATA/8/8train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            elif(prev_char == ord('9') and len(data) > 5 ):
                video.release()
                while(os.path.exists("./DATA/9/9train_{}".format(i))):
                    i += 1 
                dataframe = dataframe.loc[(dataframe.x!=0) & (dataframe.y !=0)]                    
                dataframe.to_csv("./DATA/9/9train_{}".format(i), index=False)
                for k in range(len(dataframe.values)):
                    if(k != len(dataframe.values)-1):
                        cv2.line(img, (dataframe.values[k][0],dataframe.values[k][1]), (dataframe.values[k+1][0],dataframe.values[k+1][1]), (255,255,255), 25)
                img = cv2.flip(img, 1)
                img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
                cv2.imshow("pre", img)

                i=0
                Record = False
                data = []
            #print(prev_char)
        if(Record):
            video.write(datum.cvOutputData)
           
        if c == 27: break
'''
####################################################################################################
#end = time.time()
#print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
