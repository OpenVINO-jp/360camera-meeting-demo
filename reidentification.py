##-----------------------------------
##re-identification
##
## model:
##   intel/person-reidentification-retail-0288
##   intel/facial-landmarks-35-adas-0002
##-----------------------------------

import iewrap

import cv2
import numpy as np
import time
import os

import yaml
import re   #正規表現

from scipy.spatial import distance
from munkres import Munkres               # Hungarian algorithm for ID assignment

import pprint

##person-reidentification-retail-0288

ie_reid = iewrap.ieWrapper(r'intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml', 'CPU')

##facial-landmarks-35-adas-0002
ie_faceLM   = iewrap.ieWrapper(r'intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml',   'CPU')

face_dir = r'./rsc/face/'

curr_feature = []
feature_db = []


bolDisableTimeOut = True

def PreloadImage():

    global curr_feature,feature_db

    print(f"feature_db : {len(feature_db)}")

    ## read menbers infomation from yml file
    with open(r'meeting_member.yml') as file:
        data = yaml.safe_load(file)

    ## ymlに定義したメンバーの画像を読み込んで、curr_featureに詰め込み
    ##
    ## \face_dir
    ##    \(EmployeeID)
    ##      \(ファイル) <- フォルダ内を列挙するのでファイル名は問わない

    for yaml_out in data['member']:
        print(yaml_out["EmployeeID"],yaml_out["LastName"],yaml_out["Section"],yaml_out["Position"])  # for debug

        each_face = face_dir + yaml_out["EmployeeID"]
        print(each_face) 

        if (os.path.exists(each_face)):

            for fname in os.listdir(each_face):
                img = cv2.imread(each_face + "/" + fname)
                featVec = ie_reid.blockInfer(img).reshape((256))
                pos = [0,0,0,0]

                curr_feature.append({'pos': pos,'feature': featVec, 'id': -1,'img': img, 'name': yaml_out["LastName"], 'employeeID': yaml_out["EmployeeID"], 'section': yaml_out["Section"], 'position': yaml_out["Position"], 'mouth_val': 0})

        else:
            print(f"path \"{each_face}\" is not found.")

    ## curr_featureに詰め込みが出来たら、fncReidを呼び出してfeature_dbを作る
    print(f"curr_feature : {len(curr_feature)}")
    if (len(curr_feature) > 0):
        fncReid()

    print(f"feature_db : {len(feature_db)}")

    return

def fncDrawLM(face):
    _X=0
    _Y=1

    # landmarkを全て取得
    landmark = ie_faceLM.blockInfer(face).reshape((70,)) # [1,70]
    lm=landmark[:70].reshape(35,2)  #  [[left0x, left0y], [left1x, left1y], [right0x, right0y], [right1x, right1y] ]
    #print(lm)

    # landmark
    for count in range(35):
        cv2.circle(face, (abs(int(lm[count][_X] * face.shape[1])),abs(int(lm[count][_Y] * face.shape[0]))), 2, (0,255,255), -1)

    face = cv2.circle(face,(int(lm[10][_X] * face.shape[1]),int(lm[10][_Y] * face.shape[0])), 3, (0,255,0), -1)
    face = cv2.circle(face,(int(lm[11][_X] * face.shape[1]),int(lm[11][_Y] * face.shape[0])), 3, (0,255,0), -1)

    return(face)


def fncMouthValue(person,face_ratio,face):
##上唇と下唇の差分
##
## facial-landmarks-35-adas-0002
## [Mouth] p8, p9: mouth corners on the outer boundary of the lip; p10, p11: center points along the outer boundary of the lip.
##
## person     : image
## face_ratio : ROIとface_detectのrectの割合（逆数）... イメージ全体を使う場合の考慮。face detectionの結果rectを使う場合は不要
## face       : face image
##
## faceのrect_y(face.shape[0])で割る必要あり。顔の大きさが数値に影響があるため。

    _X=0
    _Y=1

    ## landmarkを取得
    # landmark = ie_faceLM.blockInfer(person).reshape((70,)) # [1,70]
    landmark = ie_faceLM.blockInfer(face).reshape((70,)) # [1,70]
    lm=landmark[:70].reshape(35,2)  #  [[left0x, left0y], [left1x, left1y], [right0x, right0y], [right1x, right1y]... ]
    
    ##for debug
##    face = cv2.circle(face,(int(lm[10][_X] * face.shape[1]),int(lm[10][_Y] * face.shape[0])), 5, (0,0,255), -1)
##    face = cv2.circle(face,(int(lm[11][_X] * face.shape[1]),int(lm[11][_Y] * face.shape[0])), 5, (0,0,255), -1)
##    face = cv2.resize(face,(200,320))
##    cv2.imshow('face', face)
##    key = cv2.waitKey(2)

    return(abs(int((lm[10][_Y] - lm[11][_Y]) / face.shape[0] * 1000000)))
    #return(abs(int((lm[10][_Y] - lm[11][_Y]) * 1000 * face_ratio))) # personを使う場合は考慮


def CurrFeatureAppend(person,pos,face_ratio,face):
## person : detected person(cropped)
## pos    : org_posision of person (for draw rect)

    global curr_feature 
    mouth_val = fncMouthValue(person,face_ratio,face)

    featVec = ie_reid.blockInfer(person).reshape((256))
    curr_feature.append({'pos': pos, 'feature': featVec, 'id': -1,'img': person, 'name':'unknown','employeeID':'unknown', 'section':'unknown' , 'position': -1 , 'mouth_val': mouth_val })

def fncReid():
    global curr_feature, feature_db

    objid = 0
    time_out = 5                        ## how long time to retain feature vector (second()

    now = time.monotonic()

    if bolDisableTimeOut == False:    
        for feature in feature_db:
            if feature['time'] + time_out < now:
                feature_db.remove(feature)     ## discard feature vector from DB
                #print("Discarded  : id {}".format(feature['id']))

    ## If any object is registred in the db, assign registerd ID to the most similar object in the current image
    if len(feature_db)>0:
        
        #print(f"fncReid_1 feature_db : {len(feature_db)}")
        #print(feature_db)
        
        ## Create a matix of cosine distance
        cos_sim_matrix=[ [ distance.cosine(curr_feature[j]["feature"], feature_db[i]["feature"]) 
                        for j in range(len(curr_feature))] for i in range(len(feature_db)) ]
        ## solve feature matching problem by Hungarian assignment algorithm
        hangarian = Munkres()
        combination = hangarian.compute(cos_sim_matrix)

        ## assign ID to the object pairs based on assignment matrix
        for dbIdx, currIdx in combination:
            
            curr_feature[currIdx]['id'] = feature_db[dbIdx]['id']               ## assign an ID

            feature_db[dbIdx]['feature'] = curr_feature[currIdx]['feature']     ## update the feature vector in DB with the latest vector
            feature_db[dbIdx]['time'] = now                                     ## update last found time
            feature_db[dbIdx]['img'] = curr_feature[currIdx]['img']             ## cropped image
            feature_db[dbIdx]['mouth_val'] = curr_feature[currIdx]['mouth_val'] ## mouth_val
            
            curr_feature[currIdx]['name'] =  feature_db[dbIdx]['name']
            curr_feature[currIdx]['employeeID'] =  feature_db[dbIdx]['employeeID']
            curr_feature[currIdx]['section'] =  feature_db[dbIdx]['section']
            curr_feature[currIdx]['position'] = feature_db[dbIdx]['position']

    ## Register the new objects which has no ID yet
    #print("# Register the new objects which has no ID yet")
    for feature in curr_feature:
        #print(str(feature['id']))
        #if feature['id']==-1:          ## no similar objects is registred in feature_db
        if feature['id'] < 1:           ## no similar objects is registred in feature_db
            #print("# no similar objects is registred in feature_db")
            feature['id'] = objid
            feature_db.append(feature)  ## register a new feature to the db
            feature_db[-1]['time']    = now

            ## save image and info for preload data
            ## auto glow のオプションとかが必要
            #cv2.imwrite(face_dir + str(feature_db[-1]['time']) + '.jpg', feature_db[-1]['img'])            

            objid+=1

#    print(f"curr_feature : {len(curr_feature)}")
#    print(f"feature_db : {len(feature_db)}")

    return


def showPersonInfo(img):
## img : original image

    global curr_feature

    ## numbering
    for obj in curr_feature:
        id = obj['id']

## 上部に表示
#        cv2.rectangle(img, (0, 0), (320, 40),(128,0,0), -1) # 192 dark blue for frame
#        if obj['name'] != "unknown":
#            cv2.putText(img, obj['name']+"("+obj['section']+')', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        color = (128,0,0)
        if obj['name'] != "unknown":
            cv2.putText(img, obj['name']+"("+obj['section']+')', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color , 2)
            cv2.putText(img,'mouth: '+str(obj['mouth_val']), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            #print (f"{obj['position']} - {obj['name']}")

## 下部に表示
##        cv2.rectangle(img, (0, (cont_height - 20) * 2), (cont_width * 2, cont_height * 2),(128,0,0), -1) # 192 dark blue for frame
#        cv2.rectangle(img, (0, 360), (640, 400),(128,0,0), -1) # 192 dark blue for frame
#
#        if obj['name'] != "unknown":
##            cv2.putText(img, obj['name']+"(id:"+obj['employeeID']+')', (10, 195), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
#            cv2.putText(img, obj['name']+"("+obj['section']+')', (10, cont_height * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
    curr_feature.clear()

    return img,obj['position'],obj['mouth_val']


def draw_one_person(img):
## img : original image

#    global curr_feature,feature_db
    global curr_feature

    color = (255,255,255)
   
    ## check mouth val
    for obj in curr_feature[:1]:
        id = obj['id']
        cv2.rectangle(img, (0, 180), (320, 200),(128,0,0), -1) # 192 dark blue
        cv2.putText(img,'mouth: '+str(obj['mouth_val']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    curr_feature.clear()

    return img
