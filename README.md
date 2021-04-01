# lets_take_a_trip

-----------------

A US Attraction Recommender created using machine learning. For a more detailed description please read my [blog post](https://racheldilley.medium.com/us-tourist-attraction-recommender-b3753b40492a) on Medium. Watch a demo of my WebApp [here](https://youtu.be/8c6hOS2yMkw) created using Streamlit.

-----------------

### Objective:

Recommend attractions or destinations located in the contenintal United States based off of an input image.

-----------------

### Approach:

All attractions in the dataset were given labels, using natural language processing and NLP on attraction names. This was done to create attraction classes, in which to train our immage classification neural network on. Text data was preprocessed to removing punctualtion, changing to lowercase, and removing stop words for each attraction name. After intial labeling NLP, all attractions were analyzed and some hand labeling was done for mislabeled attractions. 

After attraction classes were crearted, and all images given an attraction class label, a neural network utilizing transfer learning was trained on the dataset. Feature vectors were created for each image using the pre-trained CNN VGG-16 model and these features were run through the neural network. Intial neural network results showed over fitting and class imbalance. To reduce overfitting, drop out layers were added after each dense layer and L2 reqularizers were added to each dense layer. Class combining and random undersampling was used to handle class imbalance. The remaining images in the training datset were augmented, with three variations: flip, random transform, and noise. Scoring the final NN model on the testing dataset gets an accuracy score of 0.5 and a loss score of 1.4. 

Once images are classified the model find the closest attractions to the input images and recommends these attractions to the user. This is done by calculating distances between images, using both cosine distance between a color distribution vector and a vgg-16 feature vector that was used to train the neural network. Color distribution vectors were found for each image by splitting up an image into five sections, four corner sections and a center ellipse section. For each section three vectors are found representing the color distributions of red, green, and blue. All color distribution vectors from all sections are combined into a single vector, representing an images color features. These distances were summed up, multiplying the VGG-16 vector by a scalar (as VGG-16 distances were found to be more important that color vectors for some classes), to get a total distance for each image in a class to the input image. 

Images in the class were aggregated, by attraction name, and the mean distance between the input image and all attractions in the class were found. The attractions with the shortest distance were recommended to the user.

-----------------

### Featured Techniques:

* Beautifulsoup
* Selinium
* Natural Language Processing
* Topic Modeling
* Transfer Learning with VGG-16
* Neural Network
* Cosine Distance
* Streamlit

-----------------

### Data:

Over 75,00 images scraped from [TripAdvisor](https://www.tripadvisor.com/) using Selenium and Beautifulsoup. The top 30 attractions from each state in the contental United States were included in the dataset (about 1,500 different attractions).  

