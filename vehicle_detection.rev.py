import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip

from collections import deque


from lesson_function import *
from advanced_lane_lines import *


def data_look(car_list, notcar_list):
# Define a function to return some characteristics of the dataset 
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = mpimg.imread(car_list[0]).shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = mpimg.imread(car_list[0]).dtype
    # Return data_dict
    return data_dict

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0.0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def img_features_syntheticHOG(feature_image, hog_feat1, hog_feat2, hog_feat3,
                        spatial_size=(32, 32), hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    img_features = []
    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
        elif hog_channel == 0:
            hog_features = np.hstack((hog_feat1))
        elif hog_channel == 1:
            hog_features = np.hstack((hog_feat2))
        else:
            hog_features = np.hstack((hog_feat3))
        img_features.append(hog_features)

    return np.concatenate(img_features)

def search_windows_one_hog(img, windows, clf, scaler, 
                    color_trans='RGB2RGB', xy_window=(64,64), 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    # Convert image
    feature_img = convert_image(img, color_trans)
    
    # Apply HOG once to each channel of image   
    if hog_feat:
        hog1 = get_hog_features(feature_img[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(feature_img[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(feature_img[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    on_windows = []
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(feature_img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #test_img = cv2.resize(feature_img[window[0][1]:window[1][1], window[0][0]:window[1][0]], xy_window)
        # Read out HOG data for according window
        if hog_feat:
            xpos1 = (window[0][0]//pix_per_cell)
            #dx = ((window[1][0] - window[0][0])//pix_per_cell) - cell_per_block + 1
            dx = (64//pix_per_cell) - cell_per_block + 1
            ypos1 = (window[0][1]//pix_per_cell)
            #dy = ((window[1][1] - window[0][1])//pix_per_cell) - cell_per_block + 1
            dy = (64//pix_per_cell) - cell_per_block + 1
            hog_feat1 = hog1[ypos1:ypos1+dy, xpos1:xpos1+dx].ravel()
            hog_feat2 = hog2[ypos1:ypos1+dy, xpos1:xpos1+dx].ravel() 
            hog_feat3 = hog3[ypos1:ypos1+dy, xpos1:xpos1+dx].ravel()
        else:
            hog_feat1 = []
            hog_feat2 = []
            hog_feat3 = []
        # Extract features
        features = img_features_syntheticHOG(test_img, hog_feat1=hog_feat1,
                            hog_feat2=hog_feat2, hog_feat3=hog_feat3,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    return on_windows
  
def make_the_machine(color_space='RGB', xy_window=(64,64), 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True, debug=0):
    ######
    #
    # Reading in image(s)
    #
    ######

    if debug:
        print('*** Reading in data ...')

    cars = []
    notcars = []

    images = glob.glob('Additional pics/*/*/*/*.jpeg')
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)
    images = glob.glob('Additional pics/*/*/*/*.png')
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)
            

    if 1:      
        data_info = data_look(cars, notcars)

        print('Your function returned a count of', 
            data_info["n_cars"], ' cars and', 
            data_info["n_notcars"], ' non-cars')
        print('of size: ',data_info["image_shape"], ' and data type:', 
            data_info["data_type"])
     
    if debug:
        print('*** Done.')
     
    ######
    #
    # Car, non-car feature generation
    #
    ######

    if debug:
        print('*** Sample feature generation ...')

    #
    ### Feature generation of samples for SVM training
    #
    t=time.time()
    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, 
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    if debug:
        print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

    if debug:
        print('*** Scaling ... ')
     
    #
    ### Scale features of samples
    #
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    if debug:
        print('*** Done.')
     
    ######
    #
    # Training SVM
    #
    ######

    if debug:
        print('*** Training SVM ...')

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state, shuffle=True )

    # Use a linear SVC 
    svc = LinearSVC(C=0.01)
    # Check the training time for the SVC
    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    if debug:
        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Colorspace is ', color_space, ' and HOG channel chosen:', hog_channel)
        print('Feature vector length:', len(X_train[0]))
        print('Train set:', len(X_train))
        print('Test set:', len(X_test))
        print(round(t2-t1, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        prediction = svc.predict(X_test[0:n_predict])
        print('My SVC predicts:  ', prediction)
        print('For these labels: ', y_test[0:n_predict], ' (#:', n_predict, ')')
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
        acc = accuracy_score(prediction, y_test[0:n_predict])
        print('Accuracy:', acc)
        

    if debug:
        print('*** Done.')
    
    return svc, X_scaler

def detect_in_image(img, draw_img, debug=0):    
    ######
    #
    # Detect cars with one-time only hog
    #
    ######

    if debug:
        print('*** Detecting cars ...')


    global hist_heat_map
    global clf_defined
    global svc
    global X_scaler

    #
    ### Parameter for feature generation
    #
    #color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #color_trans = 'RGB2LUV' # Due to movie format
    color_trans = 'RGB2YCrCb' # Due to movie format
    orient = 9  # HOG orientations
    #pix_per_cell = 6 # HOG pixels per cell
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    #
    ### Parameters for window sliding
    #
    x_start_stop=[None, None]
    y_start_stop = [500, 720] # Min and max in y to search in slide_window()
    xy_window=(64, 64)
    #xy_overlap=(0.75, 0.75)
    xy_overlap=(0.85, 0.85)
    sliding_areas = [[[0,1280],[400,720],(128,128)], [[0,1280],[400,640],(96,96)], [[0,1280],[410,602],(64,64)], [[160,1120],[420,580],(32,32)]]

    #
    ### Classifier Generation: Sample feature extraction, classifier training
    #
    if not(clf_defined):
        clf_defined = 1
        if 1:
            svc, X_scaler = make_the_machine(color_space=color_space, 
                                    xy_window=xy_window, spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat, debug=0)
            joblib.dump(svc, 'svc_model.pkl')
            joblib.dump(X_scaler, 'svc_xscaler.pkl')
        else:
            svc = joblib.load('svc_model.pkl')
            X_scaler = joblib.load('svc_xscaler.pkl') 
            
    #
    ### Car detection: Window sliding definition, window classification, rectangular drawing
    #
    out_img = np.copy(img)
    bbox_list = []
    for x_start_stop, y_start_stop, xy_window in sliding_areas:
        if debug:
            print(x_start_stop,':',y_start_stop,':', xy_window)

        windows = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                                xy_window=xy_window, xy_overlap=xy_overlap)
        box_list = search_windows_one_hog(img, windows, svc, X_scaler, color_trans=color_trans, 
                                xy_window=xy_window, spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)                       
        out_img = draw_boxes(out_img, box_list, color=(0, 0, 255), thick=6)
        bbox_list.extend(box_list)


    if debug:
        fig = plt.figure()
        plt.imshow(out_img)
        plt.title('Detected Windows, one-hog')

        plt.show()
     
    if debug:
        print('*** Done.')
     
    ######
    #
    # Detection and frame definition via heat map
    #
    ######

    #
    ### Heat-map generation
    #
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,bbox_list)
       
    
    
    #
    ### History of heat-map handling
    #
    # Add heat map image to the back of the historical collection
    hist_heat_map.append(heat)

    # Calculation of the weighting of historical heat-maps
    # The older the heat-map, the lower the weight
    nrmaps = min(len(hist_heat_map)+1, max_maps)
 
    # Generate overall heat-map based on actual and old heat-maps
    idxs = range(nrmaps)
    for map, weight, idx in zip(hist_heat_map, weight_map, idxs):
        hist_heat_map[idx] = map * weight

    heat = np.sum(np.array(hist_heat_map), axis=0)

    heat = apply_threshold(heat, int(4*nrmaps))

    # Transform heat map into image
    heat = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heat)
    #draw_img = draw_labeled_bboxes(np.copy(img), labels)
    draw_img = draw_labeled_bboxes(draw_img, labels)

    if debug:
         # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions, one-hog')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map, one-hog')
        fig.tight_layout()

        plt.show()
        
    return draw_img
  
def process_image(image):
    # Advanced-Lane-Line finding of P4
    lane_detected_image = lane_detection(image)
    # Use this one for only car detection
    #lane_detected_image = np.copy(image)
    # Car detection
    movie_image = detect_in_image(image, lane_detected_image, debug=0)
    
    return movie_image

# Global parameters for car detection
max_maps = 8
hist_heat_map = deque(maxlen = max_maps)
clf_defined = 0

moviepath = "./project_video.rev.mp4"
movieoutpath = "./project_video_masked.rev.mp4"
movie = VideoFileClip(moviepath)
masked_movie = movie.fl_image(process_image)
masked_movie.write_videofile(movieoutpath, audio=False)


