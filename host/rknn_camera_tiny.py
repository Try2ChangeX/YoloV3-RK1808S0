import cv2
import time
import numpy as np
from rknn_client_class import rknn_client


CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop  ","mouse    ","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


def draw(image, boxes, scores, classes, scale = 2):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        #print('class: {}, score: {}'.format(CLASSES[cl], score))
        #print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        #x *= image.shape[1]
        #y *= image.shape[0]
        #w *= image.shape[1]
        #h *= image.shape[0]

        x *= scale
        y *= scale
        w *= scale
        h *= scale


        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
       
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        
        """
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        """
       
        # print('class: {0}, score: {1:.2f}'.format(CLASSES[cl], score))
        # print('box coordinate x,y,w,h: {0}'.format(box))

if __name__ == '__main__':
    rknn = rknn_client(8002)

    IMG_SIZE = 224
   
    font = cv2.FONT_HERSHEY_SIMPLEX;
    capture = cv2.VideoCapture("data/data1.avi")
    #capture = cv2.VideoCapture(0)
    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    fps = "FPS: ??"
    while(True):
        ret, frame = capture.read()
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

            testtime=time.time()
            boxes, classes, scores, tt = rknn.inference(inputs=[image])
            testtime2=time.time()
            print("inference use time: {}", testtime2-testtime)

            testtime=time.time()
            if boxes is not None:
                draw(image, boxes, scores, classes)
                print("===boxes", boxes)
                print("===classes", classes)
                print("===scores", scores, tt)


            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time += exec_time
            curr_fps += 1
            if accum_time > 1:
                accum_time -= 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.imshow("results", image)
            c = cv2.waitKey(1) & 0xff
            if c == 27:
                cv2.destroyAllWindows()
                capture.release()
                break;
            testtime2=time.time()
            print("show image use time: {}", testtime2-testtime)
