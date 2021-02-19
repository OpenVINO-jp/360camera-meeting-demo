# from open_model_zoo_toolkit.py
# https://github.com/yas-sim/openvino_open_model_zoo_toolkit

import cv2
import numpy as np
import open_model_zoo_toolkit as omztk
from openvino.inference_engine import IENetwork, IECore
from utils.codec import CTCCodec

#import pprint

import tkinter as tk
from tkinter.scrolledtext import ScrolledText    

def text_window(text_value):

    root = tk.Tk()
    root.title("Detected Text(OpenVINO)")

    text = ScrolledText(root, font=("",15), height=20, width=50)
    text.pack()

    text.insert('1.0',text_value)
    
    root.mainloop()


def topLeftPoint(points):
# from handwritten-japanese-OCR-touch-panel-demo.py

    big_number = 1e10
    _X=0
    _Y=1
    most_left        = [big_number, big_number]
    almost_most_left = [big_number, big_number]
    most_left_idx        = -1
    almost_most_left_idx = -1

    for i, point in enumerate(points):
        px, py = point
        if most_left[_X]>px:
            if most_left[_X]<big_number:
                almost_most_left     = most_left
                almost_most_left_idx = most_left_idx
            most_left = [px, py]
            most_left_idx = i
        if almost_most_left[_X] > px and [px,py]!=most_left:
            almost_most_left = [px,py]
            almost_most_left_idx = i
    if almost_most_left[_Y]<most_left[_Y]:
        most_left     = almost_most_left
        most_left_idx = almost_most_left_idx
    return most_left_idx, most_left

def cropRotatedImage(image, points, top_left_point_idx):
# from handwritten-japanese-OCR-touch-panel-demo.py

    _X=1
    _Y=0
    _C=2
    
    point0 = points[ top_left_point_idx       ]
    point1 = points[(top_left_point_idx+1) % 4]
    point2 = points[(top_left_point_idx+2) % 4]
    
    target_size = (int(np.linalg.norm(point2-point1, ord=2)), int(np.linalg.norm(point1-point0, ord=2)), 3)

    crop = np.full(target_size, 255, np.uint8)
    
    _from = np.array([ point0, point1, point2 ], dtype=np.float32)
    _to   = np.array([ [0,0], [target_size[_X]-1, 0], [target_size[_X]-1, target_size[_Y]-1] ], dtype=np.float32)

    M    = cv2.getAffineTransform(_from, _to)
    crop = cv2.warpAffine(image, M, (target_size[_X], target_size[_Y]))

    return crop

def preprocess_input(src, height, width):
    # from handwritten-japanese-OCR-touch-panel-demo.py
    
    #src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)

    rsz = cv2.resize(src, (tw, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    outimg = np.full((height, width), 255., np.float32)
    rsz_h, rsz_w = rsz.shape
    outimg[:rsz_h, :rsz_w] = rsz
#    cv2.imshow('OCR input image', outimg)

    outimg = np.reshape(outimg, (1, height, width))
    return outimg

def get_characters(char_file):
# from handwritten-japanese-OCR-touch-panel-demo.py
    with open(char_file, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)

def outer_rect_crop(img,rects):

    tmp_x_min,tmp_y_min = 9999,9999
    tmp_x_max,tmp_y_max = 0,0
    
    for rect in rects:
        box = cv2.boxPoints(rect).astype(np.int32)

        for j in range(4):
            if (tmp_x_min > box[j][0]):
                tmp_x_min = box[j][0]

            if (tmp_y_min > box[j][1]):
                tmp_y_min = box[j][1]

            if (tmp_x_max < box[j][0]):
                tmp_x_max = box[j][0]

            if (tmp_y_max < box[j][1]):
                tmp_y_max = box[j][1]

    print(f"xmin {tmp_x_min} ymin {tmp_y_min}")
    print(f"xmax {tmp_x_max} ymax {tmp_y_max}")

    img= img[tmp_y_min:tmp_y_max,tmp_x_min:tmp_x_max].copy()
#   img = cv2.resize(img, dsize=None, fx=0.50, fy=0.50)
    return(img)

def detect_text(path):

    omz = omztk.openvino_omz()
    model = omz.textDetector()

    result_img = cv2.imread(path)
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) # 2値用

    # 閾値の設定(要調整 130-180の間程度)
    threshold = 130

    # 二値化
    #ret, img = cv2.threshold(img, threshold, 255,cv2.THRESH_OTSU) #自動で検出は期待どおりにならない
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    print(f"threshold: {ret}")

    #for debug
    #cv2.imshow('THRESH_BINARY', img)
    #key = cv2.waitKey(1)

    rects, imgs = model.run(img) # get rects from text detector
    print(rects)

    model = 'handwritten-japanese-recognition-0001'
    model = './intel/'+model+'/FP16/'+model

    ie = IECore()
    net = ie.read_network(model+'.xml', model+'.bin')

    input_blob = next(iter(net.inputs))
    out_blob   = next(iter(net.outputs))
    input_batch_size, input_channel, input_height, input_width= net.inputs[input_blob].shape
    exec_net = ie.load_network(net, 'CPU')
##

    characters = get_characters('data/kondate_nakayosi_char_list.txt')
    codec = CTCCodec(characters)

    detected_text = ""
    for i, rect in enumerate(rects):
        box = cv2.boxPoints(rect).astype(np.int32)     # Obtain rotated rectangle
 
        #OCR
        most_left_idx, most_left = topLeftPoint(box)
        crop = cropRotatedImage(img, box, most_left_idx)
        input_image = preprocess_input(crop, input_height, input_width)[None,:,:,:]

        preds = exec_net.infer(inputs={input_blob: input_image})
        preds = preds[out_blob]
        result = codec.decode(preds)
        print('OCR result ({}): {}'.format(i, result))
        detected_text = detected_text + str(result[0]) + '\n'
        cv2.polylines(result_img, [box], True, (0,255,0), 3)  # Draw bounding box

    result_img = outer_rect_crop(result_img,rects)

    cv2.imshow('Detected Rect(OpenVINO)', result_img)
    key = cv2.waitKey(1)

    if len(detected_text) != 0:
        text_window(detected_text)
