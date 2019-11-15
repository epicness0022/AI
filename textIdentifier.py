import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# restore np.load for future normal usage
np.load = np_load_old

word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}

#Below replace the numbers with the following values to prevent errors
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
value=word_index["<PAD>"], padding="post", maxlen=250)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

# #model down here(run once and comment as already have the saved model)

# model = keras.Sequential()
# #Make word vectors that are similar based on the surround words to 16 dimensions vectors
# model.add(keras.layers.Embedding(88000, 16)) 
# #To make big vectors of 16 to smaller averages
# model.add(keras.layers.GlobalAveragePooling1D())
# #Start with 16 neurons to next layer 1 neuron
# model.add(keras.layers.Dense(16, activation="relu")) 
# #One neuron with any value and makes it into 0 and 1
# model.add(keras.layers.Dense(1, activation="sigmoid"))

# model.summary()

# #Calculate the difference actual and prediction based on loss
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# x_val = train_data[:10000] #Use 10000 as the validation data
# x_train = train_data[10000:]

# y_val = train_labels[:10000] #Use 10000 as the validation data
# y_train = train_labels[10000:]

# fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)

# results = model.evaluate(test_data, test_labels)

# print(results)

# model.save("model.h5") #h5 extension save model in tensorflow

#Below is to test and utilized the trained model to see how positive is the text
model = keras.models.load_model("model.h5") # Load the saved model that has been trained

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

with open("test3.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],
        value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

# test_review = test_data[0]
# predict = model.predict([test_review])
# print("Review: ")
# print(decode_review(test_review))
# print("Prediction: " + str(predict[0]))
# print("Actual: " + str(test_labels[0]))
# print(results)
