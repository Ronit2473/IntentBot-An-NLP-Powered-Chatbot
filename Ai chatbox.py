#!/usr/bin/env python
# coding: utf-8

# In[14]:


import json
import os
import pickle
import nltk
import numpy as np
import tensorflow as tf
import random

# Load data from intents.json
with open("intents.json") as file:
    data2 = json.load(file)

try:
    if os.path.exists("data2.pickle"):
        # Load data from data2.pickle if it exists
        with open("data2.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    else:
        # If data2.pickle doesn't exist, process data from intents.json
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data2['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

        stemmer = nltk.stem.lancaster.LancasterStemmer()
        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))
        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = np.array(training)
        output = np.array(output)

        # Save processed data to data2.pickle
        with open("data2.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

except FileNotFoundError:
    print("Data file 'data2.pickle' does not exist. Please create or provide the file.")
except Exception as e:
    print(f"Error processing data: {e}")

# Define and compile the TensorFlow Keras model
input_shape = (len(training[0]),)
inputs = tf.keras.Input(shape=input_shape)
net = tf.keras.layers.Dense(8, activation='relu')(inputs)
net = tf.keras.layers.Dense(8, activation='relu')(net)
outputs = tf.keras.layers.Dense(len(output[0]), activation='softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training,output,epochs=1000,batch_size=8, verbose=1)
model.save('model.h5')

try:
    model.load("model.h5")
except:
    model.fit(training,output,epochs=1000,batch_size=8, verbose=1)
    model.save('model.h5')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Assuming bag_of_words(inp, words) returns an array with shape (1, 46)
        input_data = bag_of_words(inp, words)
        input_data = np.reshape(input_data, (1, 46))  # Reshape to match (batch_size, 46)

        results = model.predict(input_data)[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] > 0.7:
            for tg in data2["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("try again")



# In[15]:


chat()


# In[ ]:




