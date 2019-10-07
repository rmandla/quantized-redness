# Takes in a directory of images, quantizes each image using k-means X times, then measures % area of red pixels

# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
from os import listdir
from statistics import mean, stdev
from random import sample
import csv, sys

dark = [23,51,98]
light = [58,91,140]
bg_dark = [30,135,180]
bg_light = [255,255,255]

def color_quant(clusters, file, output_dir):
    # load the image and grab its width and height
    image = cv2.imread(file)
    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    
    outname = output_dir + file.split('.')[0].split('/')[-1] + '-' + str(clusters) + '-clusters.jpg'

    cv2.imwrite(outname, quant)
    
    return(outname)

def red_wing_area(file, dark, light, bg_dark, bg_light):
    
    image = cv2.imread(file)
    
    upper = np.array(light, dtype='uint8')
    lower = np.array(dark, dtype='uint8')
    
    upperbg = np.array(bg_light, dtype='uint8')
    lowerbg = np.array(bg_dark, dtype='uint8')
    
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    redpix = cv2.countNonZero(mask)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lowerbg, upperbg)
    green = cv2.countNonZero(mask)
    
    row,col = image.shape[0],image.shape[1]
    total = row*col
    
    wing = total-green
    percent = redpix / total * 100
    return percent

def measure_wing_area_pipeline(dir, output_dir, x):
    # Take in a single image, quantize it x times, then measure the percent of redness against a green background
    total = {'Indv Number':[], 'Sex':[], 'Average Redness':[], 'Std Redness':[]}
    y = 1
    listed = sample(listdir(dir),80)
    for file in listed:
        percents = []
        indv = file.split('_')[4]
        sex = file.split('_')[5].split('.')[0]
        total['Indv Number'].append(indv)
        total['Sex'].append(sex)
        for i in range(int(x)):
            percent = red_wing_area(color_quant(7, dir + file, output_dir), dark, light, bg_dark, bg_light)
            percents.append(percent)
            print(str(y) + " / " + str(len(listed)*int(x)))
            y += 1
        if int(x) > 1:
            total['Average Redness'].append(mean(percents))
            total['Std Redness'].append(stdev(percents))
        elif int(x) == 1:
            total['Average Redness'].append(percent)
            total['Std Redness'].append(None)
    with open('test-' + str(x) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(total.keys())
        writer.writerows(zip(*total.values()))
measure_wing_area_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])
    
        



