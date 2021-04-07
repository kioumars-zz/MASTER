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

path1 = os.getcwd()+"/"

path2 = os.getcwd()+"/hmdb51/"

folnew=[]

for x in range(0,len(folorg)):
    
     esm = folorg[x]+zaman
     folnew.append(esm)
     os.makedirs(path1+esm) 

video_fol_arry=[]

video_fol_arry.append((os.listdir(path2)))
video_fol_arry[0].sort()

print()
print((video_fol_arry))
print()

ejra = 0

a = 0

for x in range(0,len(video_fol_arry[0])):

     video_fol_arry.append((os.listdir(path2+video_fol_arry[0][a]+"/")))

     a+=1

print()
print((video_fol_arry))
print()

aa11 = 1 

bb11 = 0

cc11 = 0

tedadax = 16

for xx in range(0,len(video_fol_arry[0])):

     for xxx in range(0,len(video_fol_arry[aa11])):

         data = cv2.VideoCapture( path2 + str(video_fol_arry[0][bb11]) +"/"+ str(video_fol_arry[aa11][cc11]) )

         frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 

         fps = int(data.get(cv2.CAP_PROP_FPS)) 

         print()
         print("file name ==> ", path2 + str(video_fol_arry[0][bb11]) +"/"+ str(video_fol_arry[aa11][cc11]) ) 
         print("Frame ==> ", frames) 
         print("FPS ==> ", fps) 

         if  tedadax  >  frames :

             f = open(path1+"Image_error_list.txt", "a")
             f.write(path1 + str(video_fol_arry[0][bb11]) +"/"+ str(video_fol_arry[aa11][cc11]) + "\n")
             print()
             print(" >>> Your VIDEO file ( "+ str(video_fol_arry[aa11][cc11])  +" ) very small for cuting !!! <<< " )
             print()
             f.close()
             time.sleep(1)

         a = int(((frames)-3) / (tedadax+1) )
         print()
         print("taghsim bar ", tedadax ," ==>  ", a)

         b = a * (tedadax)
         print()
         print("B = ", b)

         c = (int(frames)   -   b - 3 ) 
         print()
         print("C  = ", c)

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
         print()
         print("Arry List ==> " , tedadarry)
         print()
         print("Arry Items ==> " , len(tedadarry))

         if len(tedadarry) == (tedadax+1):

             f = open(path1+folnew[1]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Imagelist.txt", "a")
             qp = 0
             for xxxx in range(0,len(tedadarry)):

                 data.set(cv2.CAP_PROP_POS_FRAMES, tedadarry[qp]   )

                 ret,frame = data.read()

                 name = path1 + folnew[0]+'/'+ str(video_fol_arry[aa11][cc11]) +'_image-' + str(tedadarry[qp]) +'.jpg'

                 print ('Creating...' + name) 

                 f.write( str(video_fol_arry[aa11][cc11]) +'_image-' +  str(tedadarry[qp]) + '.jpg' + "\n")

                 cv2.imwrite(name, frame)

                 qp += 1
 
             f.close()
    
         else:
             print()
             print(" >>> Error in cuting Video ( "+  str(video_fol_arry[aa11][cc11])   +" ) !!! <<< " ) 
             print()
             exit()


         f = open(path1+folnew[1]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Imagelist.txt", "r")
         l = [x for x in f.readlines() if x != "\n"]
         f.close()

         print(len(l))

         filearry=[]
         f = open(path1+folnew[1]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Imagelist.txt", "r")
         for xxxxx in range(0,len(l)):

             k=f.readline()[:-1]
             filearry.append((k))

         f.close()

         print()
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

             first_frame = cv2.imread(path1+folnew[0]+"/" + filearry[plp1])

             second_frame = cv2.imread(path1+folnew[0]+"/" + filearry[plp2])

             first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

             second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

             flow = cv2.calcOpticalFlowFarneback(first_frame_gray, second_frame_gray, None, 0.5, 3, 15, 3, 5, 1.5, 0)

             h, w = first_frame.shape[:2]

             stepw=w/150

             steph=h/100

             print()
             print("imgshape =>",first_frame.shape ,"\n")
             print("h =>",h ,"\n")
             print("w =>",w ,"\n")

             y, x11= np.mgrid[steph/2:h:steph, stepw/2:w:stepw].reshape(2,-1).astype(int)

             fx, fy = flow[y,x11].T

             f = open(path1 +folnew[1]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Opticalflow.txt", "a")

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
                         kkkkk=round(lll)

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
    
             plt.savefig(path1+folnew[2]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Kmeans-"+str(tedad+1)+".png")
             plt.close()

             cv2.waitKey(10)

             pop=cv2.imread(path1+folnew[2]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Kmeans-"+str(tedad+1)+".png")
             cv2.imshow('K means', pop)

             print()
             print("kmeans.labels ==>",kmeans.labels_ ,"\n")

             mydict = {i: np.where(kmeans.labels_== i)[0] for i in range(kmeans.n_clusters)}
             dictlist = []
             for key, value in mydict.items():
                 temp = [key,len(value),value]
                 dictlist.append(temp)

             print()
             print("dictlist arry ==> ",dictlist ,"\n")
             print()
             print("kmeans.cluster_centers ==>", kmeans.cluster_centers_ ,"\n")

             a=0
             jj=[]
             for xxxxxxxx in range(len( kmeans.cluster_centers_ )):
             
                 ll =(round( (math.dist( (0,0), (kmeans.cluster_centers_[a][0] , kmeans.cluster_centers_[a][1]) ) ) * 100  )  ) / 100

                 print()
                 print("ll arry ==> ",ll ,"\n")

                 if ll == 0.0:
                     ll = 0
                 jj.append(ll)
                 a += 1

             print()
             print("jj arry ==> ",jj ,"\n")

             bbb=0
             kmeansarry=[]
             for xxxxxxxxx in range(len(kmeans.cluster_centers_)):
                 kmeansarry.append([jj[bbb],dictlist[bbb][1]])
                 bbb+=1

             print()
             print("kmeansarry ==> ",kmeansarry,"\n")

             ooo = sorted(kmeansarry,key=itemgetter(0))

             print()
             print("ooo ==> ",ooo ,"\n")

             f = open(path1+folnew[1]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Linepoint.txt", "a")

             dd = 0
             for xxxxxxxxxx in range(len(ooo)):
                 f.write(str(ooo[dd][0]) + "," + str(round(ooo[dd][1])) +  "\n")
                 dd += 1

             f.close()  

             cv2.waitKey(20)

             a1 = 2
             b1 = 4
             n1 = 1

             print()
             print("ooo Lenght ==> ",len(ooo) ,"\n")

             for xxxxxxxxxxx in range(0,255):
                 imagelines = cv2.line(imagelines,(a1,a2), (b1,b2), (250-ooo[n1][1],250-ooo[n1][1],250-ooo[n1][1]), 2)
 
                 a1 += 2
                 b1 += 2
                 n1 += 1

             cv2.imshow("Periogram", imagelines)
 
             cv2.imwrite(path1+folnew[3]+"/"+ str(video_fol_arry[aa11][cc11]) +"_Periogram.jpg", imagelines)

             cv2.waitKey(20)

             oooo = np.sort(kmeansarry, axis=0)

             print()
             print("oooo ==> ",oooo ,"\n")

             print()
             print(">>>>>>>>>>>>>> ==> ",oooo[0][1] ,"\n")

             print()
             print("array avali ==> ",jj ,"\n")

             print()
             print("array avali sort ==> ",(np.sort(jj)) ,"\n")

             ooo = sorted(kmeans.cluster_centers_,key=itemgetter(1))

             print()
             print("array avali andis sort A ==> ",ooo,"\n")

             oooo = np.sort(kmeans.cluster_centers_, axis=0)

             print()
             print("array avali andis sort B ==> ",oooo,"\n")

             print()
             print("kmeans.predict ==>",kmeans.predict(mm2),"\n")
  
             lines = np.vstack([x11, y, x11+fx, y+fy]).T.reshape(-1, 2, 2)

             lines = np.int32(lines + 0.5)

             vis = cv2.cvtColor(second_frame_gray, cv2.COLOR_GRAY2BGR)

             k=cv2.polylines(vis, lines, 4, (0, 255, 0))

             for (x1, y1), (x2, y2) in lines:

                 l=cv2.circle(k, (x1, y1), 0, (0, 0, 255), -1)
  
 
             cv2.imshow('Single Frames Opticalflow', l)

             namea = path1+folnew[0]+'/'+ str(video_fol_arry[aa11][cc11]) +'_image_sabz-' + str(abc) +'.jpg'

             cv2.imwrite(namea, l) 


             cv2.waitKey(20)
             abc += 1

             a2 += 2
             b2 += 2
             plp1 +=1
             plp2 +=1
             tedad += 1

             print()
             print("TEDAD Image ==> ",  tedad)

         f = open(path1+folnew[1]+"/Linepointlist.txt", "a")
         f.write( str(video_fol_arry[aa11][cc11]) +"_Linepoint.txt" + "\n")
         f.close()

         f = open(path1+folnew[1]+"/Periogram_list.txt", "a")
         f.write( str(video_fol_arry[aa11][cc11]) +"_Periogram.jpg" +  "\n")
         f.close()  
  
         cv2.waitKey(30)

         cc11 += 1

         ejra += 1

     aa11+=1

     bb11+=1

     cc11=0     

print()
print("TEDAD Kol Video ==> ",  ejra)

print()
print(" THE END ")

data.release() 

cv2.destroyAllWindows()
