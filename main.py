import cv2

# define the yolo v3 model
from model import make_yolov3_model, WeightReader
from utills import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes

# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the probability threshold for detected objects
class_threshold = 0.6

# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def main() :
    yolov3 = make_yolov3_model()

    # load the weights
    weight_reader = WeightReader('yolov3.weights')

    # set the weights
    weight_reader.load_weights(yolov3)

    # save the model to file
    yolov3.save('model.h5')

    #upload = files.upload()

    photo_filename = 'dogs.jpeg'

    image = cv2.imread(photo_filename)

    image_h, image_w, channels = image.shape

    # define the expected input shape for the model
    input_w, input_h = 416, 416

    ##image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

    image = cv2.resize(image, (input_w, input_h))

    # make prediction
    yhat = yolov3.predict(image)
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])

    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    # suppress non-maximal boxes
    do_nms(boxes, 0.5)

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    # draw what we found
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)


if __name__ == "__main__":
    main()