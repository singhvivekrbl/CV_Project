"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters)

    detection(test_img)
    
    recognition()

    raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    
    ## To find the number of files in the directory we will use os.walk function file
#     files = next(os.walk(read_image(img_path)))[2]
#     file_count = len(files)
    
    ## iterate over number of file in the directory
    cur_count = 5 #file_count 
    for i in range(0,cur_count):   
        img = characters[i][1] # for each image we should extract it and pass it to ORB function
        img_check=cv2.ORB_create()
        key_name = img_check.detect(img,None)
        key_name, desc = img_check.compute(img, key_name)
        key_name,desc=img_check.detectAndCompute(img,None) # once the features are extracted, will saved in key_name and description list

        ## here features are stored in .txt file 
        kp_name = characters[i][0] + ".txt"
        file1 = open(kp_name, 'w')
    
        for j in range(0,np.size(desc,0)):
            file1.write(str(desc[j]))
        file1.close()
    
    return key_name,desc

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    # Connected Component Implementation / Naive Template Matching Implementation
    # Splitting of Test image into windows - 
    


    r_list = []
    row=[]

    for i in range(0, np.size(test_img,0)):     # from 0 to last row 
        for j in range (0,np.size(test_img,1)): 
            if (test_img[i][j]<240): # if any pixel in that row is <240 set flag = 0 ||| that means foreground
    
                val=0
   
            else:
                val=1
        if val==1: # its a background
            r_list.append(i)   # keeping the record of all the backgrounf rows || appending to a list 
            
    
    a = r_list[0]
    count = 1
    for i in range(0,len(r_list)):   # going to the last background row
        count = count+1
        if (a+count < r_list[i]):
            a = r_list[i]
            count =1 
            row=row+[r_list[i-1],r_list[i]]# background row under which there is an image
        row.append(i)
        


        # for column
    count=-1
    image_list=[]
    list_c=[]
    for i in range(0,len(row)):
        for j in range(0,np.size(test_img,1)):
            for k in range(row[i-1],row[i]):
                if (test_img[k][j]<200):
                    val=0

                else:
                    val=1
            if val==1:
                list_c.append(j)

    col=[]
    a = list_c[0]

    count = 1
    for i in range(0,len(r_list)):   # going to the last background row
        count = count+1
        if (a+count < r_list[i]):
            a = r_list[i]
            count =1 
            row=row+[r_list[i-1],r_list[i]] # background row under which there is an image 
        col.append(i)
        print (row)


    ## printing image
        for col in range(0,len(col)):

            w = col[col] - col[col-1] # starting column window_col[col-1]
            h = row[i]- row[i-1] # starting row window_row[line-1]
        
            crop_img = test_img[row[i-1]:row[i], col[col-1]col[col]]
        #print (len(crop_img), np.size(crop_img,0),np.size(crop_img,1))
            cropped_image = np.array(crop_img)
            cropped_image= np.reshape(crop_img, (h,w))    # to convert from 1d to 2d array
            str_name = "test_img_"+str(i)+"_"+str(col)+".jpg"
            cv2.imwrite(str_name,cropped_image)

            count = count+1
            enrollment(test_img)
            c_image = {
                    "bbox" : cropped_image,
                    "x" : col[col-1],
                    "y" : row[i-1],
                    "w" : w,
                    "h" : h,
                    "name" : "NULL"
                }

            image_list.append(c_image)

    return image_list  

def recognition():
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
