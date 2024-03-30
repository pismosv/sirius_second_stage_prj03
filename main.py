# Listing 7_14
from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt


color_index = {'bus': 'red',
               'handbag': 'steelblue',
               'giraffe': 'orange',
               'spoon': 'gray',
               'cup': 'yellow',
               'chair': 'green',
               'elephant': 'pink',
               'truck': 'indigo',
               'motorcycle': 'azure',
               'refrigerator': 'gold',
               'keyboard': 'violet',
               'cow': 'magenta',
               'mouse': 'crimson',
               'sportsball': 'raspberry',
               'horse': 'maroon',
               'cat': 'orchid',
               'boat':  'slateblue',
               'hotdog': 'navy',
               'apple': 'cobalt',
               'parkingmeter': 'aliceblue',
               'sandwich': 'skyblue',
               'skis': 'deepskyblue',
               'microwave': 'peacock',
               'knife': 'cadetblue',
               'baseballbat': 'cyan',
               'oven': 'lightcyan',
               'carrot': 'coldgrey',
               'scissors': 'seagreen',
               'sheep': 'deepgreen',
               'toothbrush': 'cobaltgreen',
               'firehydrant': 'limegreen',
               'remote': 'forestgreen',
               'bicycle': 'olivedrab',
               'toilet': 'ivory',
               'tvmonitor': 'khaki',
               'skateboard': 'palegoldenrod',
               'train': 'cornsilk',
               'zebra': 'wheat',
               'tie': 'burlywood',
               'orange': 'melon',
               'bird': 'bisque',
               'diningtable': 'chocolate',
               'hairdrier': 'sandybrown',
               'cellphone': 'sienna',
               'sink': 'coral',
               'bench': 'salmon',
               'bottle': 'brown',
               'car': 'silver',
               'bowl': 'maroon',
               'tennisracket': 'palevilotered',
               'airplane': 'lavenderblush',
               'pizza': 'hotpink',
               'umbrella': 'deeppink',
               'bear': 'plum',
               'fork': 'purple',
               'laptop': 'indigo',
               'vase': 'mediumpurple',
               'baseballglove': 'slateblue',
               'trafficlight': 'mediumblue',
               'bed': 'navy',
               'broccoli': 'royalblue',
               'backpack': 'slategray',
               'snowboard': 'skyblue',
               'kite': 'cadetblue',
               'teddy bear': 'peacock',
               'clock': 'lightcyan',
               'wineglass': 'teal',
               'frisbee': 'aquamarine',
               'donut': 'mincream',
               'suitcase': 'seagreen',
               'dog': 'springgreen',
               'banana': 'emeraldgreen',
               'person': 'honeydew',
               'surfboard': 'palegreen',
               'cake': 'sapgreen',
               'book': 'lawngreen',
               'pottedplant': 'greenyellow',
               'toaster': 'ivory',
               'stopsign': 'beige',
               'couch': 'khaki'}

resized = False


def forSecond(frame_number, output_arrays, count_arrays, average_count,
              returned_frame):
    plt.clf()
    this_colors = []
    labels = []
    sizes = []
    counter = 0

    for each_item in average_count:
        counter += 1
        labels.append(each_item + " = " + str(average_count[each_item]))
        sizes.append(average_count[each_item])
        this_colors.append(color_index[each_item])

    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(width=1000, height=500)
        resized = True

    plt.subplot(1, 2, 1)
    plt.title("Second : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True,
            startangle=140, autopct="%1.1f%%")
    plt.pause(0.01)


execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "/yolov3_.pt"
# Путь к файлам с видео
video_path_in = execution_path + "/file8.mp4"
video_path_out = execution_path + "/traffic_detected_(7_14, file8)"

video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(model_path)
video_detector.loadModel()
plt.show()

video_detector.detectObjectsFromVideo(
    input_file_path=video_path_in,
    output_file_path=video_path_out,
    frames_per_second=20,
    per_second_function=forSecond,
    minimum_percentage_probability=30,
    return_detected_frame=True,
    log_progress=True)

