from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

if __name__ == '__main__':
    # # Use the model
    #use more than 1000 epochs for actual trainning
    result = model.train(data="config.yaml", epochs=300,device = 0)  # train the model
