import streamlit as st

import pandas as pd
import numpy as np

# img load
from PIL import Image

# models
from keras.models import load_model, Model
from keras.applications import vgg16
import keras

#load model
model = load_model('../Models/vgg_cnn.h5')

#load img df
# df = pd.read_pickle('../Data/img_info_df.pkl')
# grouped = df.groupby('category')

def classify(image, vgg_model, model):
    cats = ['art', 'beaches/ocean', 'entertainment', 'gardens/zoo', 'landmarks', 'museums',
       'parks', 'sports']

    img_std = image/255

    img_vgg = get_bottleneck_features(vgg_model, img_std)
    
    predictions = np.array(model.predict(img_vgg))
    pred = np.argmax(predictions)
    
    return cats[pred] 

def histogram(image, mask, bins):
    # extract a 3D color histogram from the masked region of the image, using the supplied number of bins per channel
    hist = cv2.calcHist([image], [0,1,2], mask, [bins[0],bins[1],bins[2]],[0, 180, 0, 256, 0, 256])
    
    # normalize the histogram if we are using OpenCV 2.4
    if imutils.is_cv2():
        hist = cv2.normalize(hist).flatten()
        
    # otherwise handle for OpenCV 3+
    else:
        hist = cv2.normalize(hist, hist).flatten()

    return hist

def get_color_description(img_array, bins, color):
    img = img_array * 255
    image = cv2.cvtColor(img, color)
    
    features = []
   
    # grab the dimensions and compute the center of the image
    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    # divide the image into four rectangles/segments (top-left, top-right, bottom-right, bottom-left)
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

    # construct an elliptical mask representing the center of the image
    (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
    ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

    # loop over the segments
    for (startX, endX, startY, endY) in segments:
        # construct a mask for each corner of the image, subtracting the elliptical center from it
        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)

        # extract a color histogram from the image, then update the feature vector
        hist = histogram(image, cornerMask,bins)
        features.extend(hist)

        # extract a color histogram from the elliptical region and update the feature vector
        hist = histogram(image, ellipMask, bins)
        features.extend(hist)
    return features

def get_sorted_distances_list(df, feat):
    distances = {}
    for index, row in df.iterrows():
        pt = np.array(row[4])
        dist = np.linalg.norm(feat-pt)
        distances[index] = dist

    # distances
    sorted_dist = dict(sorted(distances.items(), key=lambda item: item[1]))
    
    return sorted_dist

def find_closest_imgs(img_class, img):
    img_df = grouped.get_group(img_class)

    bins = [8,8,8]
    color = cv2.COLOR_BGR2HSV
    feats = get_color_description(x, bins, color)

    sorted_dist = get_sorted_distances_list(img_df, feats)

    attractions = []
    urls = []
    locations = []
    while len(locations) < 3:
        for key, value in sorted_dist.items():
            if img_df.loc[key,'name'] not in attractions:
                attractions.append(img_df.loc[key,'name'])
                urls.append(img_df.loc[key,'url'])
                locations.append(img_df.loc[key,'location'])
    
    df = pd.DataFrame()
    df['name'] = attractions
    df['url'] = urls
    df['location'] = locations

    return df

def get_bottleneck_features(model, input_img):
    input_imgs = np.array([input_img])
    
    features = model.predict(input_imgs, verbose=0)
    return features

inputs = (150, 150, 3)
vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=inputs)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

#dont want model weights to change durring training
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False


st.title('US Tourist Attraction Recommender')
#st.image()
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = image.resize((150, 150)) 
    img_array = np.array(img)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Recommending...")
    label = classify(img_array, vgg_model, model)
    st.write(label)
    # closest_imgs = find_closest_imgs(label, img_array)
    # st.write(closest_imgs)



    