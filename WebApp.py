import streamlit as st
from streamlit import caching

import pandas as pd
import numpy as np
import pickle5 as pickle
import urllib.request

#location data and intialize geocoder
from geopy.geocoders import Nominatim
geocoder = Nominatim(user_agent = 'your_app_name')
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geocoder.geocode, min_delay_seconds = 1,   return_value_on_exception = None) # adding 1 second padding between calls
import requests
import urllib.parse

# img load
from PIL import Image
# from ImageFunctions import load_image_streamlit

# models
from keras.models import load_model, Model
from keras.applications import vgg16

#load cnn model
from tensorflow import keras
import os
# model = keras.models.load_model('../Models/vgg_cnn.h5')
export_path = os.path.join(os.getcwd(), 'vgg_cnn2')
model = load_model(export_path)

# color distributions
# from ImageFunctions import get_color_description, histogram
import cv2
import imutils
import sklearn.preprocessing as preprocessing

# show map
import pydeck as pdk

# create vgg model with correct input size
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

def get_color_description(img_array, bins):
    color = cv2.COLOR_BGR2HSV
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

def load_image_streamlit(url):
    ua = UserAgent()
    headers = {'user-agent': ua.random}

    response = requests.get(url, headers = headers)
    image_io = BytesIO(response.content)
    img = Image.open(image_io)    
    return img


def check_if_in_us(lat,long):
    '''
    check if latitude and longitude in US, return False if not
    '''
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
    '''
    get location object given zip code
    return list of location object, latitude, and longitude
    return empty list if not found or not in US
    '''
    try:
        loc = geocode(query={'postalcode':zip}, addressdetails=True)
        lat = float(loc.latitude)
        long = float(loc.longitude)
        if check_if_in_us(lat,long) == False: #chekc if in US
            return []
        else:
            return [loc, lat, long]
    except:
        return []

def find_lat_long(add):
    '''
    find lat and long, return empty tuple if not found or not in US
    '''
    try:
        loc = geocode(add) #try to find location using geocoder
        lat = float(loc.latitude)
        long = float(loc.longitude)
        if check_if_in_us(lat,long) == False: #check if in US
            return (None, None)
    except:
        try: 
            # try to find location using openstreetmap if geocoder fails
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(add) +'?format=json'
            response = requests.get(url).json()
            lat = float(response[0]["lat"])
            long = float(response[0]["lon"])
            if check_if_in_us(lat,long) == False: #check if in US
                return (None, None)
        except:
            # try finding coordinate using zipcode
            zip_try1 = add[-5]
            locs_try1 = get_lat_long_from_zip(zip)
            zip_try2 = add[-9:-6]
            locs_try2 = get_lat_long_from_zip(zip)
            if locs_try1 != [] and check_if_in_us(lat,long) == True: #check if in US and not empty
                lat = float(locs_try1[1])
                long = float(locs_try1[2])
            elif locs_try2 != [] and check_if_in_us(lat,long) == True: #check if in US and not empty
                lat = float(locs_try2[1])
                long = float(locs_try2[2])
            else:
                return (None, None)
    return (lat, long)

def show_map_locations(addresses, names, latitude, longitude):
    '''
    show map of recommened attraction locations and of user location
    '''
    latitude = float(latitude)
    longitude = float(longitude)

    locations = []
    lats = []
    longs = []
    name_list = []

    for idx, add in enumerate(addresses): #loop through addresses
        #find lat and long given address, write warning if not found
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
    
    # find midpoint of all coordinates
    longs = longs + [longitude]
    longs = np.array(longs)
    lats = lats + [latitude]
    lats = np.array(lats)
    mid_long  = np.average(longs) 
    mid_lat = np.average(lats)

    df = pd.DataFrame( {'lat': [latitude], 'lon': [longitude]}) 

    # create pydeck chart centered at midpoint
    # create 2 scatterlayers, one for attraction coordinates and one for user coordinate
    st.pydeck_chart(pdk.Deck(
                    # map_style='mapbox://styles/mapbox/light-v9',
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
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude= mid_lat,
                    longitude= mid_long,
                    zoom=3,
                    # pitch=30
                ),
                ))

def classify(img_vgg, model):
    '''
    find class using cnn model, using img vgg vector and return prediction
    '''
    cats = ['beaches/ocean', 'entertainment', 'gardens/zoo', 'landmarks', 'museums','parks']
    
    predictions = np.array(model.predict(img_vgg))
    pred = np.argmax(predictions) #find max value
    
    return cats[pred] 

def get_bottleneck_features(model, input_img):
    '''
    get vgg vector features of array of images
    '''
    input_imgs = np.array([input_img])
    
    features = model.predict(input_imgs, verbose=0)
    return features

def get_distance(img_feats, feats):
    '''
    get distance between vectors
    '''
    img_feats_arr = np.array(img_feats)
    pt = np.array(feats)
    return np.linalg.norm(img_feats_arr-pt)

def get_recommendations(img_class, img_array, img_vgg):
    '''
    get df of top attractions and siplay 3 images from top attractions
    '''
    # df = load_data()
    # load df with color and vgg descriptions
    file_name = img_class.replace('/', '_')
    aws_path = 'https://streamlitwebapp2.s3.us-east-2.amazonaws.com/' + file_name +'_df.pkl'
    local_path = '/Users/racheldilley/Documents/lets-take-a-trip-data/AppData/' + file_name + '_df.pkl'
    requests = urllib.request.urlopen(aws_path)
    df = pickle.load(requests)
    # df = pickle.load(open(path, 'rb'))

    #get color distribution feature vector
    bins = [8,8,8]
    img_color_des = get_color_description(img_array, bins)

    # get distances between color vectors of all imgs in class and distances between vgg vectors
    df['color_feats'] = df.apply(lambda row: get_distance(img_color_des, row[3]), axis=1)
    df['vgg_feats'] = df.apply(lambda row: get_distance(img_vgg, row[4]), axis=1)

    # df = df.astype({'name': 'category', 'location': 'category'}).dtypes

    # create color and vgg vectors and standardize 
    min_max_scaler = preprocessing.MinMaxScaler()
    color_array = df['color_feats'].values.astype(float).reshape(-1,1)
    scaled_color_array = min_max_scaler.fit_transform(color_array)
    vgg_array = df['vgg_feats'].values.astype(float).reshape(-1,1)
    scaled_vgg_array = min_max_scaler.fit_transform(vgg_array)

    # drop color and vgg columns
    df.drop(['color_feats','vgg_feats'], axis=1, inplace=True)

    # combine arrays, weighing vgg vector depending on class
    if img_class in ['gardens/zoo']:
        total_distance =  2*scaled_vgg_array + scaled_color_array
    elif img_class in ['beaches/ocean', 'landmarks', 'parks']:
        total_distance =  20*scaled_vgg_array + scaled_color_array
    else:
        total_distance =  10*scaled_vgg_array + scaled_color_array

    # add new distance column
    df['distance'] = total_distance

    # groupb attractions and find mean distance
    grouped_df = df.groupby(['name', 'location'])['distance'].mean()
    grouped_df = pd.DataFrame(grouped_df).reset_index()

    # remove attractins with no locations
    grouped_df['length'] = grouped_df.location.str.len()
    grouped_df = grouped_df[grouped_df.length > 3]

    # sort by distance ascending
    grouped_df.sort_values(by=['distance'], ascending=True, inplace=True)

    # get top 3 attractions
    top_df = grouped_df[:3].reset_index()
    atts = [top_df.loc[0,'name'], top_df.loc[1,'name'], top_df.loc[2,'name']]

    # groupp by attraction, and get groups for top 3 attractions
    grouped = df.groupby('name')
    groups = []
    for attraction in atts:
        groups.append(grouped.get_group(attraction))
    show_recommendations(groups, atts) #show recommendations

    return top_df


def show_map(df):
    '''
    display table of attractions and locations
    ask user for zip code and display map 
    '''
    names = [df.loc[0,'name'], df.loc[1,'name'], df.loc[2,'name']]
    locations = [df.loc[0,'location'], df.loc[1,'location'], df.loc[2,'location']]

    display_data = pd.DataFrame({
    'Attraction' : names,
    'Address': locations,
    })
    display_data.set_index('Attraction', inplace=True)
    st.table(display_data)

    # get user input 
    user_input = st.text_input('Enter your zip code', "")
    if st.button('Search Locations'):
        if user_input == '':
            st.write('Nothing Entered') #display if search button pressed but nothing entered
        else:
            locs = get_lat_long_from_zip(user_input) #get lat and long and display location and show map if valid input
            if locs != []:
                st.write('Locating recommended attractions relative to ' + str(locs[0].raw['display_name']))
                show_map_locations(locations, names, locs[1], locs[2])
            else:
                st.write('Did not enter valid location, or not located in contenintal Unites States')

    

def show_recommendations(groups, atts):
    '''
    show 3 images for each recommended attraction
    '''
    for idx, group in enumerate(groups):
        df = pd.DataFrame(group).reset_index()
        st.header(atts[idx])
        imgs = [df.loc[0,'url'], df.loc[2,'url'], df.loc[5,'url']]
        st.image(imgs, width = 200)


st.title('LETS TAKE A TRIP')
st.header('A US Tourist Attraction Recommendation System')
st.write('Upload an image to get some inspiration for your next vacation and input your zipcode to get an '
        'idea on how far you will be traveling for your next vacation.')


# sidebar
st.sidebar.title("With so many vacation destinations and sights to see, planning your next vactation can be overwhelming and stressful")
# st.sidebar.markdown("Have you ever wanted to recreate a past vacation in a different location?")
# st.sidebar.markdown("Or maybe you've come across some images while scrolling through social media that look like a fun destination for "
#                 "your next trip, but your not sure where it was taken")
st.sidebar.markdown("Get some vacation planning help with this application, created using a neural network with transfer learning trained "
                    "on over 70,000 tourist uploaded images scraped from Tripadvisor.")

#font size
st.markdown("""
<style>
.big-font {
    font-size:17px !important;
}
</style>
""", unsafe_allow_html=True)

# upload jpg file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # open img, resize, get arr, standardize, and get vgg features
    image = Image.open(uploaded_file)
    img = image.resize((150, 150)) 
    img_array = np.array(img)
    img_std = img_array/255
    img_vgg = get_bottleneck_features(vgg_model, img_std)

    #show uploaded img
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.markdown('<p class="big-font">Recommending... theis could take a minute</p>', unsafe_allow_html=True)
    # st.write("")

    #classify with cnn model
    label = classify(img_vgg, model)
    if label == 'entertainment':
        st.markdown(f'<p class="big-font">An {label} attraction</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="big-font">A {label} attraction</p>', unsafe_allow_html=True)
    # st.write()

    #get recommedations and show map
    df = get_recommendations(label, img_array, img_vgg)
    show_map(df)



    