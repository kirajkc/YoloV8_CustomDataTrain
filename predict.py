from ultralytics import YOLO
import cv2
# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(r"C:\Users\KARAN\Desktop\yv8\runs\detect\train\weights\last.pt")  # load a pretrained model (recommended for training)
#model1 = YOLO("yolov8n.pt")

# # Use the model
#result = model.train(data="config.yaml", epochs=5)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
#results = model("img\paper.jpg")  # predict on an image
#model1("bus1.jpg")
# read webcam
cap = cv2.VideoCapture(1)
# visualize webcam
detection_threshold = 0.7
while True:
    ret, frame = cap.read()
    results = model(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
           
            if score >= detection_threshold:
                image = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), ((255,0,0)), 2)
                cv2.putText(image, str(score), (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
    print(fps)

    cv2.imshow('frame', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# print(results)
# path = model.export(format="onnx")  # export the model to ONNX format