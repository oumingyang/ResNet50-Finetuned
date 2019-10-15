import os
import json

def get_file_path(root_path, dir_list, file_list):
    # get all of the dir and file fellow root_path
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # get dir or file path
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            #Recursive
            get_file_path(dir_file_path, dir_list, file_list)
        else:
            file_list.append(dir_file_path)


def save_file_path(file_save_path, dir_list, file_list):
    """
    file_path: dir for save file
    dir_list: list save dir_path
    file_list: list save file_path
    """
    

    # 
    for dir_path in dir_list:
        file_in_dir = "file_in_" + dir_path
        file_in_dir = []

        for file_path in file_list:
            if dir_path in file_path:
                file_in_dir.append(file_path)
            else:
                print("There is no file in:%s \n", dir_path)
        
        json_str = json.dumps(file_list)

    #
    try:
        file_save_list = open(file_save_path, "a+")
        for file in file_list:
            file_save_list.write(file)

    except IOError:
        print("Error: file operation failed!")

    else:
        print("Success: file write in success!")
        file_save_list.close()

    


if __name__ == "__main__":

    root_path = "/home/alan/work/vs_code/WorkSpace/Image_mask/Source_image"

    dir_list = []
    file_list = []
    
    get_file_path(root_path, dir_list, file_list)

    print("file_list of root path:", file_list)
    print("dir_list of root_path:", dir_list)