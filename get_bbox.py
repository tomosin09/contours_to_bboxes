import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def get_points_from_json(path_to_json):
    objects = {'filename': [], 'landmarks': [], 'plate_txt': []}
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    print('first 10 values from json\n------------------------------')
    print(data[:10])
    for i in data:
        for key, value in i.items():
            if key == 'nums':
                landmarks = []
                for list in value:
                    for k, v in list.items():
                        if k == "box":
                            landmarks.append(v)
                        if k == "text":
                            objects["plate_txt"].append(v)
                objects['landmarks'].append(landmarks)
            if key == 'file':
                objects['filename'].append(value)
    print('first 10 values from objects\n------------------------------')
    print(f'{objects["landmarks"][:10]}\n'
          f'{objects["filename"][:10]}\n'
          f'{objects["plate_txt"][:10]}')

    return objects


def to_show(images, titles):
    length = len(images)
    rows = 1
    if length < 4:
        columns = length
    else:
        rows = round(length / 2)
        columns = 3
    for i in range(len(images)):
        plt.subplot(rows, columns, i + 1);
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def get_path_to_image(path_to_dir, objects_names):
    return [os.path.join(path_to_dir, x) for x in objects_names]


def draw_landmarks(path_to_images, landmarks, len_list):
    images = []
    titles = []
    len_list = len_list if len(path_to_images) > len_list else len(path_to_images)
    for i in range(len_list):
        print(path_to_images[i])
        print(landmarks[i])
        image = cv2.imread(path_to_images[i])
        title = i
        landmark = np.array(landmarks[i])
        cv2.drawContours(image, landmark, -1, (0, 255, 0), 3)
        images.append(image)
        titles.append(title)
    return images, titles


def draw_bbox(path_to_images, bboxes, len_list):
    images = []
    titles = []
    len_list = len_list if len(path_to_images) > len_list else len(path_to_images)
    for i in range(len_list):
        print(path_to_images[i])
        print(bboxes[i])
        image = cv2.imread(path_to_images[i])
        title = i
        for bbox in bboxes[i]:
            cv2.rectangle(image, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), (0, 255, 0), 3)
        images.append(image)
        titles.append(title)
    return images, titles


def get_bbox_list(landmarks):
    bboxes = []
    for i in landmarks:
        bbox = []
        landmark = np.array(i)
        for contour in landmark:
            point_1 = np.min(contour, axis=0)
            point_2 = np.max(contour, axis=0)
            pts = [point_1.tolist(), point_2.tolist()]
            bbox.append(pts)
        bboxes.append(bbox)
    return bboxes


if __name__ == "__main__":
    json_path = '/Users/andrejilin/Desktop/saved_files/data/train.json'
    dir_dataset = '/Users/andrejilin/Desktop/saved_files/data'
    objects = get_points_from_json(json_path)
    path_to_images = get_path_to_image(dir_dataset, objects['filename'])

    # Check correct contours from objects
    # images, titles = draw_landmarks(path_to_images, objects['landmarks'], 10)
    # to_show(images, titles)

    bboxes = get_bbox_list(objects['landmarks'])
    # images_bbox, titles_bbox = draw_bbox(path_to_images, bboxes, 10)
    # to_show(images_bbox, titles_bbox)
