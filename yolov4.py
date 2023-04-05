import cv2 as cv
import pywhatkit
import pyautogui
import time
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(255, 255, 255), (0, 234, 255), (255, 255, 0),
          (255, 255, 0), (255, 0, 255), (0, 0, 255)]

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv.VideoCapture(0)

starting_time = time.time()
frame_counter = 0
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    count_1 = 0
    count_2 = 0
    for (classid, score, box) in zip(classes, scores, boxes):

        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        object_nama_1 = class_name[classid[0]]

        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        object_nama_2 = class_name[classid[0]]

        if(object_nama_1 == "person"):
            count_1 = count_1 + 1
        
        if(object_nama_2 == "chair"):
            count_2 = count_2 + 1

        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_DUPLEX, 0.3, color, 1)

    print("Jumlah Orang: ", count_1)
    print("Jumlah Kursi: ", count_2)


    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    # print(fps)
    cv.putText(frame, f'Jumlah Objek: {count_1}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)
    
    cv.putText(frame, f'Jumlah Objek: {count_2}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)
    
    cv.imshow('frame', frame)
    key = cv.waitKey(10)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
