import os
import random
from TensorflowToolbox.utility import file_io
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../data"

    file_list_dir = "../file_list/"
    data_ext = "_resize.jpg"
    label_ext = "_resize.desmap"

    cam_dir_list = file_io.get_dir_list(data_dir)
    train_list = list()
    test_list = list()
    full_file_list = list()
    for cam_dir in cam_dir_list:
        video_list = file_io.get_listfile(cam_dir, ".avi")
        
        data_list = list()
        for file_name in video_list:
            data_dir_name = file_name.replace(".avi", "")
            curr_data_list = file_io.get_listfile(data_dir_name, data_ext)

            data_list += curr_data_list


        full_file_list += [d.replace(data_ext, label_ext) + " " + d \
                        for d in data_list]

        #partition = 0.7
        #train_data_len = int(len(data_list) * partition)

        #random.shuffle(data_list)
        #train_data = data_list[:train_data_len]
        #test_data = data_list[train_data_len:]

        #train_list += [d + " " + d.replace(data_ext, label_ext) for d in train_data]
        #test_list += [d + " " + d.replace(data_ext, label_ext) for d in test_data]

    train_file_list_name = 'train_list1.txt'
    file_io.save_file(full_file_list, file_list_dir + train_file_list_name, True)

    test_file_list_name = 'test_list1.txt'
    file_io.save_file(full_file_list, file_list_dir + test_file_list_name, True)


