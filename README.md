# IntentBot-An-NLP-Powered-Chatbot
This project implements a simple chatbot using Natural Language Processing (NLP) techniques and a neural network model. The chatbot is trained on a set of predefined intents and patterns specified in a JSON file. The key components of this project include data preprocessing, model training, and an interactive chat function that utilizes the trained model to respond to user inputs.

Detailed Description
Dependencies:

nltk: Used for natural language processing tasks such as tokenization and stemming.
numpy: Used for handling arrays and numerical data.
tensorflow: Used for building and training the neural network model.
json: Used for parsing the JSON file containing the training data.
pickle: Used for saving and loading preprocessed data to avoid repeated processing.
Data Loading and Preprocessing:

The chatbot reads training data from intents.json, which contains various intents, each with corresponding patterns (user inputs) and responses.
If a preprocessed data file (data2.pickle) exists, it loads the data from this file to save time on subsequent runs.
Otherwise, it processes the data by tokenizing and stemming the words in each pattern, creating a bag-of-words model for each pattern, and generating training data for the neural network.
Model Definition and Training:

A neural network model is defined using TensorFlow's Keras API. The network consists of an input layer, two hidden layers with ReLU activation, and an output layer with softmax activation.
The model is compiled with the Adam optimizer and categorical cross-entropy loss.
If a pre-trained model (model.h5) exists, it loads the model from this file. Otherwise, it trains the model on the preprocessed data and saves the trained model to model.h5.
Chat Function:

The chat function enables user interaction with the chatbot. It takes user inputs, preprocesses them to match the format used during training, and uses the trained model to predict the intent of the input.
Based on the predicted intent, it selects an appropriate response from the training data and outputs it to the user.
Key Files
intents.json: Contains the training data with various intents, patterns, and responses.
data2.pickle: Stores preprocessed data to avoid repeated processing on subsequent runs.
model.h5: Stores the trained neural network model.
