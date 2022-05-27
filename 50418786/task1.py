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
from xml.dom.pulldom import CHARACTERS
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

    #raise NotImplementedError

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
    
    # iterate over number of file in the directory
    cur_count = 5 #file_count 
    for i in range(0,cur_count):   
        img = characters[i][1] # for each image we should extract it and pass it to ORB function
        img_check=cv2.ORB_create()
        key_name = img_check.detect(img,None)
        key_name, desc = img_check.compute(img, key_name)
        key_name,desc=img_check.detectAndCompute(img,None) # once the features are extracted, will saved in key_name and description list
          
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

    try:
        ## Reading the giving image using orb detector
        img_binary = cv2.threshold(test_img, 128, 255, cv2.THRESH_BINARY)[1]
        orb = cv2.ORB_create()
        kp = orb.detect(img_binary,None)
        kp, des = orb.compute(img_binary, kp)
       
        ## retrieving the shape of descriptor
        row, col = des.shape
        
        r_list = []
        row1=[]
        
        ## Extracting row
        for i in range(0, col):  
            
            for j in range (0,row): 
#                 print(i,j)
                thres = 250
                if test_img[i][j]<thres: # if any pixel in that row is <250 set flag = 0 ||| that means foreground
                    val=0
                    r_list.append(val)
                    
                else:
                    val=1
                    r_list.append(val) 
#                 print(r_list)
        a = r_list[0]
        count = 1
        for i in range(0,len(r_list)):   # going to the last background row
            
            count = count+1
            
            check_val = a+count < r_list[i]
            if check_val:
                a = r_list[i]
                count =1 
                row1=row1+[r_list[i-1],r_list[i]]# background row under which there is an image
            row1.append(i)
            
        # Extracting column   
        count=0
        image_list=[]
        list_c=[]
        for i in range(0,len(row1)):
            for j in range(0,col):
                print(j)
                for k in range(row1[i-1],row1[i]):
                    
                    thres = 250
                    if test_img[i][j]<thres:
                        val=0
                        #list_c.append(val)
                    else:
                        val=1
                        #list_c.append(val)
                if val==1:
                    list_c.append(j)
#                     print(list_c)

        col=[]
        a = list_c[0]

        count = 1
        for i in range(0,len(r_list)):   # going to the last background row
            count = count+1
            check_val= a+count < r_list[i]
            if check_val:
                a = r_list[i]
                count =1 
                row1=row1+[r_list[i-1],r_list[i]] # background row under which there is an image 
            col.append(i)
            
            for col in range(0,len(col)):
           
                crop_img = test_img[row1[i-1]:row1[i], col[col-1]:col[col]]
                cropped_image = np.array(crop_img)
                                
                image_list.append(cropped_image)
            return image_list

    except AttributeError:
        print("shape not found")

   

     

def recognition():
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    # key_feat, descrip = enrollment(characters)
    # detect = detection(test_img)


    # list_des = []
    # list_dete = []
    # final= []
    # for i in descrip:
    #     list_des.append(i)
        
    #     for j in detect:
    #         list_dete.append(j)
    #         check = np.allclose(list_des,list_dete)
    #         if check == True:
    #             final = "FOUND"
    #         else:
    #             final = "UNKNOWN"
    # return final
    #raise NotImplementedError


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
    
    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=True)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
