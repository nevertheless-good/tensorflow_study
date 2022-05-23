# https://www.tensorflow.org/tutorials/keras/text_classification

# Import Module
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


# Download IMDB Dataset
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# Check Dataset
# print("Sample: {}, Label: {}".format(len(train_data), len(train_labels)))
# print(train_data[0])
# print(len(train_data[0]), len(train_data[1]))

# Dictionary (Index - Word)
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0]))


# Convert to Tensor (Not one-hot encoding, add pad)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# print(len(train_data[0]), len(train_data[1]))
# print(train_data[0])


# Model
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None, )))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['accuracy'])



# Test Set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# Train
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)


# Graph
history_dict = history.history
print(history_dict.keys())


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']


epochs = range(1, len(acc) + 1)


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validataion loss')
plt.title('Training and Validataion loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validataion acc')
plt.title('Training and Validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
