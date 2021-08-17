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
            start_point = (int(bbox[0][0]), int(bbox[0][1]))
            end_point = bbox[0][0] + bbox[1], bbox[0][1]+bbox[2]
            cv2.rectangle(image, start_point,
                          end_point, (0, 255, 0), 3)
        images.append(image)
        titles.append(title)
    return images, titles


def get_bbox_list(landmarks):
    bboxes = []
    for i in landmarks:
        bbox = []
        landmark = np.array(i)
        for contour in landmark:
            start_point = np.min(contour, axis=0)
            end_point = np.max(contour, axis=0)
            width = end_point[0] - start_point[0]
            height = end_point[1] - start_point[1]
            pts = [start_point.tolist(), width, height]
            bbox.append(pts)
        bboxes.append(bbox)
    return bboxes

def get_data_wider_face(filenames, bboxes, landmarks):
    wider_data = {'names':[], 'data':[]}
    if len(filenames) == len(bboxes) and len(filenames) == len(landmarks):
        print('------------------------------\n'
              'lists names, objects and landmargs is equal'
              '\n------------------------------')
        for i in range(len(filenames)):
            wider_data['names'].append(filenames[i])
            data = []
            for b, l in zip(bboxes[i], landmarks[i]):
                data.extend([[b[0][0], b[0][1], b[1], b[2],
                             l[0][0], l[0][1], l[1][0],
                             l[1][1], l[2][0], l[2][1], l[3][0], l[3][1]]])
            wider_data['data'].append(data)
        print(f'wider_name is {wider_data["names"][:5]}\n'
              f'and data {wider_data["data"][:5]}')
        return wider_data
    else:
        print('------------------------------\n'
              'lists names, objects and landmargs is not equal, make sure the data has not been lost'
              '\n------------------------------')
        return None

def create_wider_label(names, points):
    f = open('label.txt', 'w')
    for name, pts in zip(names, points):
        f.write(f'# {name}\n')
        for pt in pts:
            for val in pt:
                f.write(f'{val} ')
            f.write('\n')


if __name__ == "__main__":
    json_path = '/Users/andrejilin/Desktop/saved_files/data/train.json'
    dir_dataset = '/Users/andrejilin/Desktop/saved_files/data'
    objects = get_points_from_json(json_path)
    path_to_images = get_path_to_image(dir_dataset, objects['filename'])

    # Check correct contours from objects
    # images, titles = draw_landmarks(path_to_images, objects['landmarks'], 10)
    # to_show(images, titles)

    bboxes = get_bbox_list(objects['landmarks'])

    # Check correct bboxes
    # images_bbox, titles_bbox = draw_bbox(path_to_images, bboxes, 10)
    # to_show(images_bbox, titles_bbox)

    objects['bboxes'] = bboxes
    wider_data = get_data_wider_face(objects['filename'], objects['bboxes'], objects['landmarks'])

    create_wider_label(wider_data['names'], wider_data['data'])




