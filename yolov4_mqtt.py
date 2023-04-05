#Sebagai Dasar Fungsi IoT
import paho.mqtt.client as mqtt #Library mqtt
client = mqtt.Client('lens_DLM467CCS5lJqFq1JVUAe3qG5kG') #broker publik yang gratis
client.connect('broker.emqx.io', 1883) #broker publik yang gratis dari EMQX IO

#Library OpenCV untuk Yolov4
import cv2 as cv 
import time
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(255, 255, 255), (0, 234, 255), (255, 255, 0),
          (255, 255, 0), (255, 0, 255), (0, 0, 255)]

#Perintah untuk membaca Library Class atau Sample data AI dari Yolo.
class_name = []
with open('classes.txt', 'r') as f: #Dengan File Class bersama classes.txt
    class_name = [cname.strip() for cname in f.readlines()]
# Menampilkan Nama Class
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#Mengatur Penggunaan kamera
cap = cv.VideoCapture('https://192.168.8.179:8080/video') # IP Address didapatkan dari kamera yang terhubung ke jaringan.
#cap = cv.VideoCapture(0) Untuk Webcam Laptop
#cap = cv.VideoCapture(1) Untuk Webcam External

starting_time = time.time()
frame_counter = 0 
while True:
    ret, frame = cap.read()
    frame_counter += 1 #Ketika objecknya terdeteksi maka akan dijumlahkan
    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    count_1 = 0 #Settting Awal perhitungan objek yang ditentukan mulai dari 0
    for (classid, score, box) in zip(classes, scores, boxes):

        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid[0]], score)
        object_nama_1 = class_name[classid[0]]

        if(object_nama_1 == "laptop"): # Menentukan Objeck yang akan di hitung berdasarkan hasil baca kamera, "Laptop" dapat diganti dengan benda lain.
            count_1 = count_1 + 1 # Perintah menjumlahkan jumlah objek ang terdeteksi.
    

        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_DUPLEX, 0.3, color, 1)

    print("Jumlah Laptop: ", count_1) #Menampilkan hasil perhitungan pada Console
    client.publish("/abdulaziz/yolov4mqtt/kursi", count_1) # Mengirim data ke Publish Mqtt
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime #Perhitungan untuk menghasilkan FPS kamera
    print(fps) #Menampilkan FPS kamera pada Console
    cv.putText(frame, f'FPS: {fps}', (20, 50), 
                cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2) #Menampilkan FPS Kamera pada View Kamera.
    cv.putText(frame, f'Objek: {count_1}', (20, 70), 
                cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2) #Menampilkan Jumlah Objek pada View Kamera.
    cv.imshow('frame', frame)
    key = cv.waitKey(10)
    if key == ord('q'): #Key untuk Close dari View Kamera
        break
cap.release()
cv.destroyAllWindows()