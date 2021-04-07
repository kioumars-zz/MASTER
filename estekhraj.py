#from __future__ import print_function

import os 
import sys 
import csv
import cv2
import math
import time
import shutil
import pickle
import datetime
import itertools
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib.image as mpimg

from math import log
from sklearn import datasets
from operator import itemgetter
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter, defaultdict
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_blobs, make_circles, make_moons


np.set_printoptions(threshold=np.inf, linewidth=np.nan)

folorg = ["image_","doc_","kmeans_","periogram_"]

zaman = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

path = os.getcwd()+"/hmdb51/shake_hands/"


folnew=[]


if not "video" in os.listdir(path):

     print("VIDEO Folder NOT Found !!")
     exit()

else:
    if not len(os.listdir(path+"/video/")) :

             print("VIDEO File NOT Found in VIDEO Folder !!")
             exit() 



for x in range(0,len(folorg)):
    
     esm = folorg[x]+zaman
     folnew.append(esm)
     os.makedirs(path+esm) 

"""

file1 = pathlib.Path("images")

if  file1.exists():

     shutil.rmtree('images')
     os.makedirs("images"+zaman) 

 

"""   

gl = 0

videoarry = os.listdir(path+"/video/")

for xx in range(0,len(videoarry)):
  

     video_file_name = videoarry[gl]


     data = cv2.VideoCapture(path + "/video/" + video_file_name )

     tedadax=16

     frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 

     print("Frame = ", frames) 

     fps = int(data.get(cv2.CAP_PROP_FPS)) 
  
     print("FPS = ", fps) 

     if tedadax  >  frames :

         print("\n") 
         print()
         print(" >>> Your VIDEO file ( "+ video_file_name  +" ) very smal for cuting !!! <<< " ) 
         print("\n") 
         print()
         exit()


     a = int(((frames)-3) / (tedadax+1) )

     print("taghsim bar ", tedadax ," ==>  ", a)

     b = a * (tedadax)

     print("B = ", b)

     c = (int(frames)   -   b - 3 ) 

     print("C  = ", c)

     #time.sleep(1)

     tedadarry=[2] 

     d = 0

     for xxx in range(0,(tedadax)):
         if c > 0 :
             if (int(c/b) < 1):
                 d += ( a + 1 ) 
                 tedadarry.append(d+1)
                 c -= 1
         else:
             d += ( a ) 
             tedadarry.append(d+1)

     print("Arry List ==> " , tedadarry)
     print("Arry Items ==> " , len(tedadarry))
     #time.sleep(1)

     if len(tedadarry) == (tedadax+1):

         f = open(path+"/"+folnew[1]+"/"+videoarry[gl]+"_Imagelist.txt", "a")
         qp = 0
         for xxxx in range(0,len(tedadarry)):

             data.set(cv2.CAP_PROP_POS_FRAMES, tedadarry[qp]   )

             ret,frame = data.read()

             name = path + '/'+folnew[0]+'/'+videoarry[gl]+'_image-' + str(tedadarry[qp]) +'.jpg'

             print ('Creating...' + name) 

             f.write(videoarry[gl]+'_image-' +  str(tedadarry[qp]) + '.jpg' + "\n")

             cv2.imwrite(name, frame)

             qp += 1
             #time.sleep(1)

         f.close()
    
     else:
         print("\n") 
         print()
         print(" >>> Error in cuting Video ( "+ video_file_name  +" ) !!! <<< " ) 
         print("\n") 
         print()
         exit()


     f = open(path+"/"+folnew[1]+"/"+videoarry[gl]+"_Imagelist.txt", "r")
     l = [x for x in f.readlines() if x != "\n"]
     f.close()

     print(len(l))

     #cv2.waitKey(100)

     filearry=[]
     f = open(path+"/"+folnew[1]+"/"+videoarry[gl]+"_Imagelist.txt", "r")
     for xxxxx in range(0,len(l)):

         k=f.readline()[:-1]
         filearry.append((k))

     f.close()

     #cv2.waitKey(100)

     print(filearry)

     tedad=0

     plp1=0

     plp2=1
     
     c1 = 36

     a2=2

     abc=1

     b2=2

     imagelines = 255 * np.ones((c1,516,3), np.uint8)


     for xxxxxx in range(0,len(l)-1):


         first_frame = cv2.imread(path  + "/"+folnew[0]+"/" + filearry[plp1])

         second_frame = cv2.imread(path + "/"+folnew[0]+"/" + filearry[plp2])

         first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

         second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

         flow = cv2.calcOpticalFlowFarneback(first_frame_gray, second_frame_gray, None, 0.5, 3, 15, 3, 5, 1.5, 0)

         h, w = first_frame.shape[:2]


         stepw=w/150

         steph=h/100

         print("imgshape =>",first_frame.shape ,"\n")
         print("h =>",h ,"\n")
         print("w =>",w ,"\n")

         y, x11= np.mgrid[steph/2:h:steph, stepw/2:w:stepw].reshape(2,-1).astype(int)


         fx, fy = flow[y,x11].T


         f = open(path   + "/"+folnew[1]+"/"+videoarry[gl]+"_Opticalflow.txt", "a")

         i = 0

         matrix2 = []



         for xxxxxxx in range(len(x11)):
        
             z =round( (   math.atan2(-(fy[i]), -(fx[i]))    *  (  180  /  math.pi  )  ) )

             if z < 0:
                     z += 360 

             zx=((round(fx[i]*100))/100)
             zy=((round(fy[i]*100))/100)

             ff = (math.dist( (x11[i],y[i]),  ((x11[i]+zx) , (y[i]+zy)))) 

             pp = ((round(ff * 100  ) )/100) 
             
             if pp > 1.5 :
                 
                 if z != 0:
                     lll=math.log(z)
                     #print(lll)

                     kkkkk=round(lll)
                     #print(kkkkk)
                 else:    
                      kkkkk =0
                 matrix2.append([(pp),(kkkkk)])

             else:
                 z=0
                 pp=0
                 matrix2.append([(pp),(z)])

             

             f.write( str(x11[i]) + "," + str(y[i]) + "," +  str(x11[i]+zx) + "," + str(y[i]+zy)  + "," + str(zx) + "," + str(zy)  + "," +      str(pp)   + "," + str(z)  +  "\n")


             i += 1


         f.close()  

         #cv2.waitKey(100)

         mm2= np.float32(matrix2)


         kmeans = KMeans(algorithm='elkan', copy_x=True, init='k-means++', max_iter=300,
          n_clusters=256, n_init=10, n_jobs=1, precompute_distances='auto',
          random_state=None, tol=0.0001, verbose=0)
         kmeans.fit(mm2)
         plt.scatter(mm2[:,0],mm2[:,1],c=kmeans.labels_, cmap='rainbow')
         plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], marker='*' ,s=200, color='black')
         plt.title ('Nemoodar')
         plt.xlabel('Tool')
         plt.ylabel('Zavie')

         plt.savefig(path +"/"+folnew[2]+"/"+videoarry[gl]+"_Kmeans-"+str(tedad+1)+".png")
         #plt.show()
         plt.close()

         cv2.waitKey(20)

         pop=cv2.imread(path +"/"+folnew[2]+"/"+videoarry[gl]+"_Kmeans-"+str(tedad+1)+".png")
         cv2.imshow('K means', pop)

         print("kmeans.labels ==>",kmeans.labels_ ,"\n")

         mydict = {i: np.where(kmeans.labels_== i)[0] for i in range(kmeans.n_clusters)}
         dictlist = []
         for key, value in mydict.items():
             temp = [key,len(value),value]
             dictlist.append(temp)

         print("dictlist arry ==> ",dictlist ,"\n")
         print()
         print("kmeans.cluster_centers ==>",kmeans.cluster_centers_ ,"\n")

         a=0
         jj=[]
         for xxxxxxxx in range(len(kmeans.cluster_centers_)):
             
             ll =(round( (math.dist( (0,0), (kmeans.cluster_centers_[a][0] , kmeans.cluster_centers_[a][1]) ) ) * 100  )  ) / 100
             if ll == 0.0:
                 ll = 0
             jj.append(ll)
             a += 1

         print("jj arry ==> ",jj ,"\n")

         bbb=0
         kmeansarry=[]
         for xxxxxxxxx in range(len(kmeans.cluster_centers_)):
             kmeansarry.append([jj[bbb],dictlist[bbb][1]])
             bbb+=1

         print("kmeansarry ==> ",kmeansarry,"\n")


         ooo = sorted(kmeansarry,key=itemgetter(0))

         print("ooo ==> ",ooo ,"\n")

         f = open(path +"/"+folnew[1]+"/"+videoarry[gl]+"_Linepoint.txt", "a")

         dd = 0
         for xxxxxxxxxx in range(len(ooo)):
             f.write(str(ooo[dd][0]) + "," + str(round(ooo[dd][1])) +  "\n")
             dd += 1

         f.close()  

         cv2.waitKey(20)

         a1 = 2
         b1 = 4
         n1 = 1

         print("ooo Lenght ==> ",len(ooo) ,"\n")


         for xxxxxxxxxxx in range(0,255):
             imagelines = cv2.line(imagelines,(a1,a2), (b1,b2), (250-ooo[n1][1],250-ooo[n1][1],250-ooo[n1][1]), 2)
             #imagelines = cv2.line(imagelines,(a1,a2), (b1,b2), (ooo[n1][1],ooo[n1][1],ooo[n1][1]), 2)

             a1 += 2
             b1 += 2
             n1 += 1

         cv2.imshow("Periogram", imagelines)

         cv2.imwrite(path +"/"+folnew[3]+"/"+videoarry[gl]+"_Periogram.jpg", imagelines)

         cv2.waitKey(20)


         oooo = np.sort(kmeansarry, axis=0)

         print("oooo ==> ",oooo ,"\n")

         print(">>>>>>>>>>>>>> ==> ",oooo[0][1] ,"\n")


         print("array avali ==> ",jj ,"\n")

         print("array avali sort ==> ",(np.sort(jj)) ,"\n")

         ooo = sorted(kmeans.cluster_centers_,key=itemgetter(1))

         print("array avali andis sort A ==> ",ooo,"\n")


         oooo = np.sort(kmeans.cluster_centers_, axis=0)

         print("array avali andis sort B ==> ",oooo,"\n")


         print("kmeans.predict ==>",kmeans.predict(mm2),"\n")


         lines = np.vstack([x11, y, x11+fx, y+fy]).T.reshape(-1, 2, 2)


         #cv2.waitKey(100)


         lines = np.int32(lines + 0.5)


         vis = cv2.cvtColor(second_frame_gray, cv2.COLOR_GRAY2BGR)


         k=cv2.polylines(vis, lines, 4, (0, 255, 0))

         for (x1, y1), (x2, y2) in lines:

             l=cv2.circle(k, (x1, y1), 0, (0, 0, 255), -1)


         #doubleframes = cv2.addWeighted(first_frame,0.4,l,0.9,0)
  
         #cv2.imshow('Double Frames Opticalflow', doubleframes)


         cv2.imshow('Single Frames Opticalflow', l)

         namea = path  + '/'+folnew[0]+'/'+videoarry[gl]+'_image_sabz-' + str(abc) +'.jpg'

         cv2.imwrite(namea, l) 


         cv2.waitKey(20)
         abc += 1

         a2 += 2
         b2 += 2
         plp1 +=1
         plp2 +=1
         tedad += 1
         print("TEDAD Ejra ==> ",  tedad)

     f = open(path +"/"+folnew[1]+"/Linepointlist.txt", "a")
     f.write(videoarry[gl]+"_Linepoint.txt" + "\n")
     f.close()

     f = open(path +"/"+folnew[1]+"/Periogram_list.txt", "a")
     f.write(videoarry[gl]+"_Periogram.jpg" +  "\n")
     f.close()  

     #cv2.waitKey(100)

     gl += 1
     cv2.waitKey(50)

#print("TEDAD >>>>> ",  tedad)
#cv2.waitKey(0)

print(" THE END ")

data.release() 

cv2.destroyAllWindows()
