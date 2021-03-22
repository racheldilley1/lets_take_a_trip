import streamlit as st

import pandas as pd
import numpy as np
import pickle

#location data
from geopy.geocoders import Nominatim
geocoder = Nominatim(user_agent = 'your_app_name')
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geocoder.geocode, min_delay_seconds = 1,   return_value_on_exception = None) 
# adding 1 second padding between calls
import requests
import urllib.parse

# img load
from PIL import Image

from ImageFunctions import load_image_streamlit

# models
from keras.models import load_model, Model
from keras.applications import vgg16
import keras

#load model
model = load_model('../Models/vgg_cnn.h5')

# color distributions
import cv2
import imutils

import sklearn.preprocessing as preprocessing


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

# show map
import pydeck as pdk

def check_if_in_us(lat,long):
    top = 49.3457868 # north lat
    left = -124.7844079 # west long
    right = -66.9513812 # east long
    bottom =  24.7433195 # south lat
    if lat > top or lat < bottom:
        return False
    elif long > right or long < left:
        return False
    else:
        return True

def get_lat_long_from_zip(zip):
    try:
        loc = geocode(query={'postalcode':zip}, addressdetails=True)
        lat = float(loc.latitude)
        long = float(loc.longitude)
        if check_if_in_us(lat,long) == False:
            return []
        else:
            return [loc, lat, long]
    except:
        return []

def find_lat_long(add):
    try:
        loc = geocode(add)
        lat = float(loc.latitude)
        long = float(loc.longitude)
        if check_if_in_us(lat,long) == False:
            return (None, None)
    except:
        try: 
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(add) +'?format=json'

            response = requests.get(url).json()
            lat = float(response[0]["lat"])
            long = float(response[0]["lon"])
            if check_if_in_us(lat,long) == False:
                return (None, None)
        except:
            zip_try1 = add[-5]
            locs_try1 = get_lat_long_from_zip(zip)
            zip_try2 = add[-9:-6]
            locs_try2 = get_lat_long_from_zip(zip)
            if locs_try1 != []:
                lat = float(locs_try1[1])
                long = float(locs_try1[2])
            elif locs_try2 != []:
                lat = float(locs_try2[1])
                long = float(locs_try2[2])
            else:
                return (None, None)
    return (lat, long)

def show_map_locations(addresses, names, latitude, longitude):
    latitude = float(latitude)
    longitude = float(longitude)

    # after initiating geocoder
    locations = []
    lats = []
    longs = []
    name_list = []

    for idx, add in enumerate(addresses):
        
        lat, long = find_lat_long(add)
        if lat != None or long != None:
            lats.append(lat)
            longs.append(long)
            name_list.append(names[idx])
            locations.append(add)
        else:
            st.write('Coudnt find geolocation for ' + names[idx])

    
    data = pd.DataFrame({
    'Recommened Attraction' : name_list,
    'Address': locations,
    'lat' : lats,
    'lon' : longs
    })

    st.write(data)
    
    # Adding code so we can have map default to the center of the data
    longs = longs + [longitude]
    longs = np.array(longs)
    lats = lats + [latitude]
    lats = np.array(lats)
    midpoint = (np.average(longs), np.average(lats))

    st.write(midpoint)


    # df = pd.DataFrame( {'lat': [latitude], 
    #                     'lon': [longitude]})
    df = pd.DataFrame( {'lat': [midpoint[0]], 'lon': [midpoint[1]]})
      
    
    st.write(df)
    

    st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                    latitude= midpoint[0],
                    longitude= midpoint[1],
                    zoom=2
                    ),
                    layers=[
                        pdk.Layer(
                        'HexagonLayer',
                        data=data,
                        get_position='[lon, lat]',
                        elevation_scale=50,
                        elevation_range=[0, 1000],
                        pickable=True,
                        extruded=True,
                        ),
                    pdk.Layer(
                    'ScatterplotLayer',
                    data=data,
                    opacity=0.9,
                    stroked=True,
                    filled=True,
                    radius_scale=10,
                    radius_min_pixels=10,
                    radius_max_pixels=60,
                    line_width_min_pixels=3,
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    # zoom=3,
                    ), pdk.Layer(
                    'ScatterplotLayer',
                    data=df,
                    opacity=0.9,
                    stroked=True,
                    filled=True,
                    radius_scale=10,
                    radius_min_pixels=10,
                    radius_max_pixels=60,
                    line_width_min_pixels=3,
                    get_position='[lon, lat]',
                    get_color='[120,140]',
                    # zoom=3,
                    ),
                ],
                ))

def classify(img_vgg, model):
    cats = ['beaches/ocean', 'entertainment', 'gardens/zoo', 'landmarks', 'museums',
       'parks']
    
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

def get_bottleneck_features(model, input_img):
    input_imgs = np.array([input_img])
    
    features = model.predict(input_imgs, verbose=0)
    return features

def get_distance(img_feats, feats):
    img_feats_arr = np.array(img_feats)
    pt = np.array(feats)
    return np.linalg.norm(img_feats_arr-pt)

def get_recommendations(img_class, img_array, img_vgg):
    #load df with color descriptions
    file_name = img_class.replace('/', '_')
    path = '/Users/racheldilley/Documents/lets-take-a-trip-data/AppData/' + file_name + '_df.pkl'
    df = pickle.load(open(path, 'rb'))
    st.write(df.head())

    bins = [8,8,8]
    color = cv2.COLOR_BGR2HSV
    img_color_des = get_color_description(img_array, bins, color)
    df['color_feats'] = df.apply(lambda row: get_distance(img_color_des, row[5]), axis=1)
    df['vgg_feats'] = df.apply(lambda row: get_distance(img_vgg, row[6]), axis=1)

    # df = df.astype({'name': 'category', 'location': 'category'}).dtypes

    min_max_scaler = preprocessing.MinMaxScaler()
    color_array = df['color_feats'].values.astype(float).reshape(-1,1)
    scaled_color_array = min_max_scaler.fit_transform(color_array)

    vgg_array = df['vgg_feats'].values.astype(float).reshape(-1,1)
    scaled_vgg_array = min_max_scaler.fit_transform(vgg_array)

    df.drop(['color_feats','vgg_feats'], axis=1, inplace=True)

    total_distance =  3*scaled_vgg_array + scaled_color_array
    df['distance'] = total_distance

    grouped_df = df.groupby(['name', 'location'])['distance'].mean()
    grouped_df = pd.DataFrame(grouped_df).reset_index()

    # remove attractins with wrong locations
    grouped_df['length'] = grouped_df.location.str.len()
    grouped_df = grouped_df[grouped_df.length > 3]

    grouped_df.sort_values(by=['distance'], ascending=True, inplace=True)

    top_df = grouped_df[:3].reset_index()
    atts = [top_df.loc[0,'name'], top_df.loc[1,'name'], top_df.loc[2,'name']]

    grouped = df.groupby('name')
    groups = []
    for attraction in atts:
        groups.append(grouped.get_group(attraction))
    show_recommendations(groups, atts)

    return top_df


def show_map(df):

    names = [df.loc[0,'name'], df.loc[1,'name'], df.loc[2,'name']]
    locations = [df.loc[0,'location'], df.loc[1,'location'], df.loc[2,'location']]

    display_data = pd.DataFrame({
    'Attraction' : names,
    'Address': locations,
    })
    st.table(display_data)


    user_input = st.text_input('Enter your zip code', "")
    if st.button('Search Locations'):
        if user_input == '':
            st.write('Nothing Entered')
        else:
            locs = get_lat_long_from_zip(user_input)
            if locs != []:
                st.write('Locating recommended attractions relative to ' + str(locs[0].raw['display_name']))
                show_map_locations(locations, names, locs[1], locs[2])
            else:
                st.write('Did not enter valid location, or not located in contenintal Unites States')

    

def show_recommendations(groups, atts):
    for idx, group in enumerate(groups):
        df = pd.DataFrame(group).reset_index()
        st.write(atts[idx])
        imgs = [df.loc[0,'url'], df.loc[2,'url'], df.loc[5,'url']]
        st.image(imgs, width = 200, height= 200)

st.title('US Tourist Attraction Recommender')

# upload jpg file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # open img, resize, get arr, standardize, and get vgg features
    image = Image.open(uploaded_file)
    img = image.resize((150, 150)) 
    img_array = np.array(img)
    img_std = img_array/255
    img_vgg = get_bottleneck_features(vgg_model, img_std)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Recommending...")

    label = classify(img_vgg, model)
    st.write('A ' + label + ' attraction')

    df = get_recommendations(label, img_array, img_vgg)
    show_map(df)



    