import cv2
import numpy as np

## window mode 1:
## meeting window
def meeting_window(im_count, im_list, im_orign, posZoomPerson, interpolation=cv2.INTER_LINEAR):

    try:
        posZoomIndex = np.roll([0,1,2,3], posZoomPerson * -1)

        ## numbering
        #for i in range(im_count):
        #    cv2.putText(im_list[posZoomIndex[i]],str(posZoomIndex[i]+1), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        ## A area(first member)
        forcus_area = cv2.resize(im_list[posZoomIndex[0]], (960,600))

        ## B area(members 2-4) インデックスに対応した画像をリサイズ
        im_list_resize = [cv2.resize(im_list[zi], (320,200), interpolation)
                          for zi in posZoomIndex[1:4]]

        members_area = cv2.vconcat(im_list_resize)
        cv2.rectangle(members_area,(0,0), (0,600), (0,0,0), 10) #frame line

        ## A+B w:1280 x h:600
        up_area = cv2.hconcat([forcus_area, members_area])
        
        im_bottom = cv2.resize(im_orign[0],(1280,200))
        cv2.line(im_bottom,(0,0), (1280,0), (0,0,0), 10) #frame line

        return cv2.vconcat([up_area,im_bottom])

    except:
        #編集時にエラーになった場合は元動画を返却する
        print("error occured at meeting_window")

        for im in im_list:
            print(f"im : {im.shape[0]},{im.shape[1]}")
        
        return cv2.resize(im_orign[0],(1280,800))

## window mode 2:
## meeting window
## 2 x 2
def two_two_window(im_count, im_list, im_orign, interpolation=cv2.INTER_LINEAR):

    try:
        im_list_resize = [cv2.resize(im, (640,400), interpolation)
                          for im in im_list[0:4]]

        ## numbering
        ## 番号付けが必要な場合

        #[cv2.putText(img,str(i+1), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2) for i,img in enumerate(im_list_resize)]

        #for i in range(im_count):
        #    cv2.putText(im_list[i],str(i+1), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        up_area     = cv2.hconcat(im_list_resize[0:2])
        bottom_area = cv2.hconcat(im_list_resize[2:4])
        rtn_area = cv2.vconcat([up_area,bottom_area])
        cv2.rectangle(rtn_area,(640,0), (640,800), (0,0,0), 5) #frame line
        cv2.rectangle(rtn_area,(0,400), (1280,400), (0,0,0), 5) #frame line                
        return rtn_area

    except:
        #編集時にエラーになった場合は元動画を返却する
        print("error occured at two_two_window")
        return cv2.resize(im_orign[0],(1280,800))
