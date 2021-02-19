import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from google.cloud import vision
import io

import cv2

def text_window(text_value):

    root = tk.Tk()
    root.title("Detected Text(Google Cloud Vision)")

    text = ScrolledText(root, font=("",15), height=20, width=50)
    text.pack()

    text.insert('1.0',text_value)
    
    root.mainloop()


def rectViewer(path,texts):

    im_result = cv2.imread(path)

    for text in texts:
#       print('\n"{}"'.format(text.description))

        x_min = text.bounding_poly.vertices[0].x
        y_min = text.bounding_poly.vertices[0].y
        x_max = text.bounding_poly.vertices[2].x
        y_max = text.bounding_poly.vertices[2].y

        cv2.rectangle(im_result,(x_min,y_min),(x_max,y_max), (128,0,0), 3) # 128 dark blue    

    img_resize = cv2.resize(im_result, dsize=None, fx=0.50, fy=0.50)
    cv2.imshow('Detected Rect(Google Cloud Vision)', img_resize)
    key = cv2.waitKey(1)


# [START vision_text_detection]
def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_text_detection]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints =["ja-t-i0-handwrit"])
    response = client.text_detection(image=image, image_context=image_context)
    #response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    texts = response.text_annotations

    if len(texts) != 0:
        rectViewer(path,texts)
        text_window(texts[0].description)

    #return(texts)

##    print('Texts:')
##
##    for text in texts:
##        print('\n"{}"'.format(text.description))
##
##        vertices = (['({},{})'.format(vertex.x, vertex.y)
##                    for vertex in text.bounding_poly.vertices])
##
##        print('bounds: {}'.format(','.join(vertices)))
##


    # [END vision_python_migration_text_detection]
# [END vision_text_detection]


