import os


def get_list_data(path_to_dir):
    dirs = os.listdir(path_to_dir)
    images_names = []
    for i in dirs:
        list_images = os.listdir(os.path.join(path_to_dir, i))
        for l in list_images:
            images_names.append(os.path.join(i, l))
    return images_names

def create_val_txt(names):
    f = open('val.txt', 'w')
    for name in names:
        f.write(f'/{name}\n')


if __name__ == '__main__':
    path = '/Users/andrejilin/Desktop/saved_files/widerface/val/images'
    names = get_list_data(path)
    create_val_txt(names)
