from ultralytics import YOLO


def run():        
    model = YOLO('yolov8m.pt')

    model.train(data='test.yaml', epochs=50)

if __name__ == '__main__':
    run()
# results =model.predict(source='C:\\Users\\OMEN\\Desktop\\ai\\test',save=True)