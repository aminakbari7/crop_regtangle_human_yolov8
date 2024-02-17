from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")
img=cv2.imread("00.jpg")
results = model(img, classes=0)
clss = results[0].boxes.cls.cpu().tolist()
boxess = results[0].boxes.xyxy.cpu().tolist()
i=0
for box, clss in zip(boxess, clss):
             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]),int(box[3]) ), (15*i, 30*i,24*i), 2)
             i+=1
cv2.imshow("ttt",img)
cv2.waitKey()