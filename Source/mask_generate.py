import os

def load_file():
    #Get current file path
    current_path = os.path.abspath(__file__)

    #Get current file father_dir
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Source_Image")
    mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Mask_Image")
    
    #Path of image for train
    fzm_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"fzm")
    cl_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"cl")
    fzm_img = os.path.join(os.path.abspath(fzm_image_path + os.path.sep),"fangzuming0.jpg")
    fzm_mosaic_path = os.path.join(os.path.abspath(mosaic_image_path + os.path.sep),"fzm")
    fzm_mosaic = os.path.join(os.path.abspath(fzm_mosaic_path + os.path.sep),"fzm0_mosaic.jpg")


    print("Current_path:" + current_path)
    print("fzm_image_path:" + fzm_image_path)
    print("fzm_image:" + fzm_img)
    print("cl_image_path:" + cl_image_path)

    print("fzm_mosaic_path:" + fzm_mosaic_path)
    print("fzm_mosaic:" + fzm_mosaic)

    for i in os.walk(fzm_image_path):
        print(i[2])

load_file()
