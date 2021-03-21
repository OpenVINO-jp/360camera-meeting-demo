## 360 camera Meeting demo
##
## openvino ir model:
##
##   intel/face-detection-adas-0001
##   intel/face-detection-retail-0005
##   intel/pedestrian-detection-adas-0002
##   intel/person-detection-retail-0013
##
## keyboard:
##
##   t   ... toggle window mode.
##           starting -> meeting -> 2x2 -> starting ..
##
##   1-4 ... zoom person at meeting mode.
##           "1" key is pushed, zoom to no.1 person
##
##   w   ... zoom white-board
##   i   ... reading white-board by OpenVINO
##   g   ... reading white-board by CloudVisionAPI
##
##   p   ... show person detect
##   f   ... show face detect
##   r   ... show detected(croped) rect
##
##   l   ... show face-landmark

import iewrap
import tile_resize as f_tr
import reidentification as f_reid

import cv2
import numpy as np
import time

#import asyncio
import text_detect_gcp as gcp_d
import text_detect_OpenVINO as openvino_d

from operator import itemgetter
from scipy import stats
import pprint

baseImg = {}## frame
imgBuf = {} ## for infer
areaUD = {} ## "u" or "d" or "m"
#label  = []
flgWindow_mode = 1        ## 0:starting 1:meeting view 2:2x2 view 3:white-board

bShowDetectedPersonRect = False   ## True or False
bShowDetectedFaceRect = False     ## True or False
bShowDetectedRect = False         ## True or False
bShowDetectedFaceLM = False       ## True or False

posZoomPerson = 0         ## for zooming person number
flgWiteboard_mode = False ## white board view

## confidence
## ここの閾値は動画に応じて調整
# jikosyoukai:0.6 / 0.2

pd_confidence = 0.3
fd_confidence = 0.4

## PersonDetect を強制する場合はtrue
forcePersonDetect = False

## FaceDetect を強制する場合はtrue
## forcePersonDetect=trueの場合はforcePersonDetect優先
forceFaceDetect = True

#cap = cv2.VideoCapture(r'./rsc/mov/speaker.mov')
cap = cv2.VideoCapture(r'./rsc/mov/jikosyoukai.mov')

##face-detection-adas-0001
#ie_fd   = iewrap.ieWrapper(r'intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml','CPU',10)

##face-detection-retail-0005
ie_fd   = iewrap.ieWrapper(r'intel/face-detection-retail-0005/FP16/face-detection-retail-0005.xml','CPU',10)

##pedestrian-detection-adas-0002
#ie_pd   = iewrap.ieWrapper(r'intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml','CPU',10)

##person-detection-retail-0013
ie_pd   = iewrap.ieWrapper(r'intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml','CPU',10)


# detect temp path
detect_temp_path = 'rsc/test.png'

# dummy image
im_blank = cv2.imread('rsc/blank.jpg')

## --- gcp OCR ---
## async def handler(func,*args):
##    return func(*args)

#async def callback_gcp_detect(target_image):
def gcp_detect(target_image):
    cv2.imwrite(detect_temp_path,target_image)
    gcp_d.detect_text(detect_temp_path)

## --- gcp OCR ---


## --- OpenVINO OCR ---
#async def callDetectOpenVINO(target_image):
def openVINO_detect(target_image):
    cv2.imwrite(detect_temp_path,target_image)
    openvino_d.detect_text(detect_temp_path)

## --- OpenVINO OCR ---


## --- 話者特定 ---
mouth_db = [] # 過去の履歴
tmp_calc = [0,0,0,0]
mouth_score = []
calc_time = time.monotonic()

def fncWhoisSpeaker(position,mouth_val,current_pos):
#def fncWhoisSpeaker(position,mouth_val):
## 戻り値はズームをする人のpos
## 左上、右上、左下、右下の順に、0,1,2,3

    global mouth_db,tmp_calc,mouth_score,calc_time
    tmp_val = []
    time_out = 2 ## タイムアウトを設ける場合。秒数指定

    #print(f"position :{position} mouth_val :{mouth_val}") ## for debug

    ## 4人そろっていない場合は抜けている分に0を入れる
    for i in range(4):
        if i not in position:
            position += [i]
            mouth_val += [0]

    ## 前回の取り出し
    if (len(mouth_db) > 0):
        tmp_val = mouth_db.pop(0)
    else:
        tmp_val = [[0,1,2,3],[0,0,0,0]]

    #print(f"tmp_val:{tmp_val}") # for debug

    mouth_db.append([position,mouth_val])

    if (len(tmp_val) > 0):
        for i, pos in enumerate(position):
            #print(f"pos {pos} i {i}")
            if (pos != -1):
                tmp_calc[pos] = abs(tmp_val[1][pos] - mouth_val[pos])

    #print(f"tmp_calc :{tmp_calc}") # for debug
    #print(f"mouth_db  :{mouth_db}") # for debug

    mode = 0

    if (len(tmp_calc) > 0):
        speaker = tmp_calc.index(max(tmp_calc))
        mouth_score += [speaker]
        print(f"time: {speaker} mouth score: {mouth_score}")

        if len(mouth_score) > 10: ## 直近(0-n)フレーム分の中から最頻値と，その数を返す
            mode, count = stats.mode(mouth_score)
            print(f"mode: {int(mode)} count: {int(count)}")
            mouth_score.pop(0)

    #return(int(mode))

    ## タイムアウト考慮。タイムアウトに未達の場合は現在のカメラポジションを返却
    if time.monotonic() > calc_time + time_out:
        print("***")
        calc_time = time.monotonic()
        return(int(mode))
    else:
        return(current_pos)


## --- 話者特定 ---


def callback(infId, output):
    global baseImg, imgBuf, areaUD, flgWindow_mode,posZoomPerson,flgWiteboard_mode, bShowDetectedRect,bShowDetectedPersonRect,bShowDetectedFaceRect,bShowDetectedFaceLM
    
    ## Draw bounding boxes and labels onto the image
    output = output.reshape((200,7))# [1,1,200,7]
    img = imgBuf.pop(infId)
    baseImg_tmp = baseImg.pop(infId)

    fArea = areaUD.pop(infId) ## "u" or "d" or "m"
    img_h, img_w, _ = img.shape

    person_count = 0    
    croped_person = []
    croped_person_fd = []
    croped_person_pd = []

    crop_rate = 2.0
    content_width = int(160 * crop_rate)
    content_height = int(100 * crop_rate)

    face_ratio = int(content_height * 2 / 200) ## ROI内における人・顔の割合。person detect採用時はy軸切り出し固定(200px)なので初期値に入れておく

    ## person detect
    for obj in output:
        imgid, clsid, confidence, x1, y1, x2, y2 = obj
        
        if confidence>pd_confidence:  ## Draw a bounding box when confidence
            
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)

            ##if (x1 > 800) : # 検出したくない領域を検出した場合（遠隔地用のディスプレイなど）の判定
            #    print("display area detected")
            #    break

            if (bShowDetectedPersonRect == True):
                color = (255,0,0) ## blue
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) ## for rect check

            person = img[y1:y2,x1:x2].copy() ## Crop for face detect
            person_count += 1

            ## カメラ設置位置、テーブルの大きさに応じて、切り出す領域の調整が必要
            ## sample ... 320 x 200 にするために検出した中心位置から切り出し
            
            ## --------------------------------------------------
            ## person detectを切り出しの画像として使用する場合
            grav_x = int((x1 + x2) / 2)
            grav_y = int((y1 + y2) / 2)

            ## cropする範囲が画面の外の場合            
            if grav_x < 160:
                grav_x = 160

            #print(f"person detect grav {grav_y},{grav_x}")
           
            person_tmp = img[y1:(y1 + 200),(grav_x - 160):(grav_x + 160)].copy()

            ## 画面右下にディスプレイ（遠隔地）がある想定
            if (grav_x > 1500 and grav_y > 800):
                color = (0,0,255) ## red
                #print(f"red : {grav_x,grav_y}")
                cv2.rectangle(img, (grav_x - 160, y1), (grav_x + 160, y1 + 200), color, 2) ## for rect check
                break
            
            croped_person_pd += [(person_tmp,grav_x,grav_y,face_ratio,person)]

            ## --------------------------------------------------
            ##  person -> face detect
            fd_det = ie_fd.blockInfer(person).reshape((200,7))   

            for fd in fd_det:
                #[ image_id, label, conf, xmin, ymin, xmax, ymax ] ##face-detection-adas-0001

                if fd[2] > fd_confidence:  ## Draw a bounding box when confidence>0.3

                    xmin = abs(int(fd[3] * person.shape[1]) + x1)
                    ymin = abs(int(fd[4] * person.shape[0]) + y1)
                    xmax = abs(int(fd[5] * person.shape[1]) + x1)
                    ymax = abs(int(fd[6] * person.shape[0]) + y1)

                    face_ratio = content_height * 2 / abs(int(ymax - ymin))
                    face_img = img[ymin:ymax,xmin:xmax].copy() ## Crop for face landmark

                    if (bShowDetectedFaceRect == True):
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2) ## for rect check

                    if (bShowDetectedFaceLM == True):
                        img[ymin:ymax,xmin:xmax] = f_reid.fncDrawLM(face_img)

                    ## --------------------------------------------------
                    ## face detectを切り出しの画像として使用する場合

                    if forcePersonDetect != True:
                        grav_x = int((xmin + xmax) / 2)
                        grav_y = int((ymin + ymax) / 2)

                        ## cropする範囲が画面の外の場合
                        if grav_x < content_width:
                            grav_x = content_width

                        if grav_y < content_height:
                            grav_y = content_height

                        #print(f"face detect grav {grav_y},{grav_x}")
                        person_tmp = img[(grav_y - content_height):(grav_y + content_height),(grav_x - content_width):(grav_x + content_width)].copy()

                        if (bShowDetectedRect == True):
                            cv2.rectangle(img, (grav_x - content_width, grav_y - content_height), (grav_x + content_width, grav_y + content_height), (0,0,255), 2) ## for rect check              

                        croped_person_fd += [(person_tmp,grav_x,grav_y,face_ratio,face_img)]

## --------------------------------------------------

    ## person or face 多く検出した方を採用
    ## forcePersonDetect,forcePersonDetectのtrue,falseでも制御                        
    ## 中心位置grav_y, grav_xでソートする場合(画面上部、画面下部の順) ... itemgetter(2,1)
    ## 中心位置grav_xでソートする場合(分割検出の場合など。左右の順) ... itemgetter(1)
                        
    if forcePersonDetect:
        #print("person-detection is forced")
        croped_person = sorted(croped_person_pd, key=itemgetter(2,1)) 

    elif forceFaceDetect:
        #print("face-detection is forced")
        croped_person = sorted(croped_person_fd, key=itemgetter(2,1))

    elif (len(croped_person_fd) <= len(croped_person_pd)):
        #print("person-detection is selected")
        croped_person = sorted(croped_person_pd, key=itemgetter(2,1))

    else:
        #print("face-detection is selected")
        croped_person = sorted(croped_person_fd, key=itemgetter(2,1))


    ## 同じ人の複数回検出(近しいエリア)を判定して削除
    ## list内全て

    dist = 50 ## 指定px以内にrectが存在する場合は重複と判定する
    croped_person_del_index = [] ## 重複検出用
    i = 0
    while (i < len(croped_person) - 1):
        j = i + 1
        while (j < len(croped_person)):
            if (abs(croped_person[i][1] - croped_person[j][1]) < dist and abs(croped_person[i][2] - croped_person[j][2]) < dist):
                croped_person_del_index += [i]

            j+=1
        i+=1

    ## 重複領域を消込
    if (len(croped_person_del_index) > 0):
        for i,cp_d_i in enumerate(croped_person_del_index):
            del croped_person[cp_d_i-i]

    ## for debug            
    #for i,cp in enumerate(croped_person):
    #    print (f"{i} - {cp[1],cp[2],cp[3]}")

    tmp_face_ratio = [row[3] for row in croped_person] ## face_ratio
    tmp_face_img = [row[4] for row in croped_person]   ## face imag for landmark 
    croped_person = [row[0] for row in croped_person]  ## 画像のみ

    ##re-id
    position = [] # reid出来た人
    mouth_val = []
    for i,cp in enumerate(croped_person):
        f_reid.CurrFeatureAppend(cp,(0,0,0,0),tmp_face_ratio[i],tmp_face_img[i])
        f_reid.fncReid() # 個人認識
        cp,pos,mouth = f_reid.showPersonInfo(cp) # cropした領域に所属情報を表示
        position += [pos]
        mouth_val += [mouth]

    ##話者特定
    #if (len(position) >= 4): ## n 人以上の場合（検出精度が低くブランクが頻出する場合）
    posZoomPerson = fncWhoisSpeaker(position,mouth_val,posZoomPerson)

    ##ここで元の領域に戻す
    if (fArea == "m"):
        baseImg_tmp[0:1080,200:1600] = cv2.resize(img,(1400,1080))
        img = baseImg_tmp

##    if fArea == "u":
##        baseImg_tmp[0:540,370:1330] = cv2.resize(img,(960,540))
##    else:
##        baseImg_tmp[540:1080,370:1330] = cv2.resize(img,(960,540))        
##    img = baseImg_tmp


    ##画面が全体映しではなく、かつ、2人以上検出した場合
    ##2分割(u/d)で処理する場合は変更が必要

    if (flgWindow_mode == 1 or flgWindow_mode == 2):
        if(len(croped_person) >= 2):
            #print(f"pd:{len(croped_person_pd)} fd:{len(croped_person_fd)}")

            tmp_croped_person_count = len(croped_person)
            if (tmp_croped_person_count > 4):
                tmp_croped_person_count = 4

            ## insert blank ブランク画像を入れる(水増し)
            while len(croped_person) < 4:
                croped_person += [im_blank]

            ## toggle layout
            ## 5人以上の動画の場合、4人までを編集する
            ## posZoomPerson ... 現在拡大表示を行っている人
        
            if flgWindow_mode == 1:
                im_tile_resize = f_tr.meeting_window(tmp_croped_person_count,croped_person[:4],[img],posZoomPerson)
            elif flgWindow_mode == 2:
                im_tile_resize  = f_tr.two_two_window(tmp_croped_person_count,croped_person[:4],[img])

        else:
            ## 検出結果が1人の場合は大写しにする
            print("croped_person is one")
            im_tile_resize = cv2.resize(img,(1280,800))
            im_tile_resize = f_reid.draw_one_person(im_tile_resize)

            cv2.putText(im_tile_resize,"Only one person is detected.", (25,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

    elif(flgWiteboard_mode == True):
        ##ホワイトボード
        cv2.rectangle(img, (510,114), (1214,510), (255,255,255), 2) ## for rect check of white-board
        img = img[114:510,510:1214].copy() ## Crop for face detect
        im_tile_resize = cv2.resize(img,(1280,800))
        cv2.putText(im_tile_resize,"White-board", (25,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

    else:
        ##元画像を表示
        im_tile_resize = cv2.resize(img,(1280,800))


    cv2.imshow('Meeing', im_tile_resize)

    key = cv2.waitKey(1)

    if key == ord("t"):
        ## toggle 0 -> 1 -> 2 -> 0
        flgWindow_mode += 1
        if flgWindow_mode > 2:
            flgWindow_mode = 0

    elif key == ord("w"):
        if flgWiteboard_mode == True:
            flgWiteboard_mode = False
        else:
            flgWiteboard_mode = True

    elif key == ord("g"):
        ##Google CloudVisionAPI
        #asyncio.run(callback_gcp_detect(img))
        gcp_detect(img)
        
    elif key == ord("i"):
        ##intel OpenVINO
        #asyncio.run(callDetectOpenVINO(img))
        openVINO_detect(img)

    elif key == ord("r"):
        if (bShowDetectedRect == True):
            bShowDetectedRect = False
        else:
            bShowDetectedRect = True

    elif key == ord("p"):
        if (bShowDetectedPersonRect == True):
            bShowDetectedPersonRect = False
        else:
            bShowDetectedPersonRect = True

    elif key == ord("f"):
        if (bShowDetectedFaceRect == True):
            bShowDetectedFaceRect = False
        else:
            bShowDetectedFaceRect = True

    elif key == ord("l"):
        if (bShowDetectedFaceLM == True):
            bShowDetectedFaceLM = False
        else:
            bShowDetectedFaceLM = True

    elif key == ord("1"):
        posZoomPerson = 0
        
    elif key == ord("2"):
        posZoomPerson = 1
        
    elif key == ord("3"):
        posZoomPerson = 2
        
    elif key == ord("4"):
        posZoomPerson = 3

def main():
    global baseImg, imgBuf, areaUD, flgWindow_mode
    
    ie_pd.setCallback(callback)

    f_reid.PreloadImage() ## preload files for build db

    while True:
        ret, img = cap.read()
        if ret==False:
            break

        #cv2.rectangle(img, (510,114), (1214,510), (255,255,255), 2) ## for rect check of white-board
        #cv2.rectangle(img, (200,0), (1500,1080), (255,255,255), 2)  ## for rect check

## 全体を利用
##        refId = ie_pd.asyncInfer(img)   ## Inference
##        imgBuf[refId]=img
##        baseImg[refId]=img

## 中央部分を検出用に利用
        img_middle = img[0:1080,200:1600].copy() # Crop for face detect
        refId = ie_pd.asyncInfer(img_middle)     ## Inference
        imgBuf[refId]=img_middle
        baseImg[refId]=img
        areaUD[refId]="m"

## 上下を分けて検出(マージ処理の実装が必要)
##
##        #上部検出
##        img_u = img[0:540,370:1330].copy() # Crop for face detect
##        refId = ie_pd.asyncInfer(img_u)   ## Inference
##        baseImg[refId]=img
##        imgBuf[refId]=img_u
##        areaUD[refId]="u"
##
##        #下部検出
##        img_d = img[540:1080,370:1330].copy() # Crop for face detect
##        refId = ie_pd.asyncInfer(img_d)   ## Inference
##        baseImg[refId]=img
##        imgBuf[refId]=img_d
##        areaUD[refId]="d"

        #time.sleep(1/30)

if __name__ == '__main__':
    main()
