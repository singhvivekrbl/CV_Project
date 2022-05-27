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
from typing import List
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
    #img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    # Using SOBEL Filters for Edge Detection
    print (characters)
    for i in range(0,5):    
        img = characters[i][1]
        sift=cv2.xfeatures2d.SIFT_create()
        kp,desc=sift.detectAndCompute(img,None)
        kp_name = "sift_keypoints_" + characters[i][0] + ".txt"
        file1 = open(kp_name, 'w')
        if (str(kp)=='()' and str(desc) == 'None'):
            file1.write(str(img))
        else: 
            #print (np.size(desc,0),np.size(desc,1))
            for j in range(0,np.size(desc,0)):
                file1.write(str(desc[j]))
        file1.close()
    #print (characters[0][0], characters[1][0], characters[2][0], characters[3][0],characters[4][0])

    #raise NotImplementedError

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
    #print (np.size(test_img,0),np.size(test_img,1))
    # 
    print (" row: ", np.size(test_img,0), " Col: ", np.size(test_img,1))
    #print (test_img.dtype)
    window_r = ()

    for r in range(0, np.size(test_img,0)):     # from 0 to last row 
        flag=1
        for c in range (0,np.size(test_img,1)): 
            if (test_img[r][c]<240): # if any pixel in that row is <240 set flag = 0 ||| that means foreground
                flag=0
                break
            else:
                flag=1
        if (flag==1): # its a background
            window_r = window_r + (r,)   # keeping the record of all the backgrounf rows || appending to a list 
            #print (window_r)
            #print (len(window_r))
    window_row=()
    min = window_r[0] # 1st blank row | background row
    #print (" Window r : ")
    #print (window_r)
    count = 1
    for i in range(0,len(window_r)):   # going to the last background row
        count = count+1
        if (min+count < window_r[i]):
            min = window_r[i]
            count =1 
            window_row=window_row+(window_r[i-1],window_r[i]) # background row under which there is an image 
            #print (window_r[i])
    window_row=window_row+(window_r[i],)
    #print (window_row)
    window_c=()
    line=1
    count=-1
    confusion_matrix = [[]]
    while (line <= len(window_row)):
        del window_c          
        window_c=()
        for c in range(0,np.size(test_img,1)):
            flag_c=1
            for r in range(window_row[line-1],window_row[line]):
                if (test_img[r][c]<200):
                    flag_c=0
                    break
                else:
                    flag_c=1
            if (flag_c==1):
                window_c = window_c+(c,)
        window_col=()
        min = window_c[0]
        count_i=1
        for i in range (0, len(window_c)):
            count_i = count_i+1
            if (min+count_i < window_c[i]):
                min = window_c[i]
                count_i=1
                window_col=window_col+(window_c[i-1],window_c[i])
        window_col = window_col+(window_c[i],)
        # check for col wise whitespace done
    
        #print (window_col)
        col=1
        while (col < len(window_col)):
            w = window_col[col] - window_col[col-1]
            h= window_row[line]- window_row[line-1]
            crop_img = test_img[window_row[line-1]:window_row[line], window_col[col-1]:window_col[col]]
            #print (len(crop_img), np.size(crop_img,0),np.size(crop_img,1))
            arr_crop_img = np.array(crop_img)
            arr_crop_img= np.reshape(crop_img, (h,w))    # to convert from 1d to 2d array
            str_name = "test_img_"+str(line)+"_"+str(col)+".jpg"
            cv2.imwrite(str_name,arr_crop_img)
            count = count+1
            sift=cv2.xfeatures2d.SIFT_create()
            kp,desc=sift.detectAndCompute(arr_crop_img,None)
            if (str(desc)=='None'):
                # Then the descriptor is a point 
                print ("template Matching for Point")
            else:
                # Then store in confusion matrix 
                print (np.size(desc,1))
                for j in range (0,np.size(desc,1)):
                    #print (str (desc[j]))
                    print ("Confusion Matrix [", count,"][",j,"]:")
                    #confusion_matrix[count].append(desc[j])
                    #print (confusion_matrix[count][j])
                print (" SIFT FEATURE EXTRACTED!!!!")
            col=col+2
        line = line +2
    #print (arr_np)

    #raise NotImplementedError

def recognition():
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

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
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=True)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
