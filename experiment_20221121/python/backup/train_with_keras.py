import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
import matplotlib.pyplot as plt

import util

EPOCH = 50

def get_NNModel(lr=1e-3):
    InputTensor = tf.keras.Input(shape=(38,))
    H1 = layers.Dense(20, activation='relu')(InputTensor)
    H2 = layers.Dense(10, activation='relu')(H1)
    Output = layers.Dense(1, activation='softmax')(H2)

    model = tf.keras.Model(inputs = InputTensor, outputs = Output, name = "NNModel")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[metrics.AUC(), metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()]
    )
    return model

def train_with_NNModel(train_dataset, val_dataset, test_dataset):
    model = get_NNModel()
    history = model.fit(train_dataset[0],
                    train_dataset[1],
                    batch_size=5,
                    epochs=EPOCH,
                    validation_data=val_dataset)

    print("============")
    scores = model.evaluate(test_dataset[0], test_dataset[1])
    print("Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

    plot_history(history)

def get_simple_sequential_model():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid'),
        ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def train_with_simple_sequential_model(train_dataset, test_dataset):
    model = get_simple_sequential_model()
    history = model.fit(train_dataset[0], train_dataset[1], batch_size=10, epochs=EPOCH)

    print("============")
    model.evaluate(test_dataset[0],  test_dataset[1], verbose=2)
    plot_history(history)

def get_model_summary(model):
    print(model.summary())

def plot_history(train_history):
    plt.plot(train_history.history['loss'])  
    # plt.plot(train_history.history['val_loss'])
    plt.title('Train History')  
    plt.ylabel('loss')  
    plt.xlabel('Epoch')  
    # plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.legend(['loss'], loc='upper left')  
    plt.show()

if __name__ == '__main__':
    # train_dataset, val_dataset, test_dataset = util.split_train_val_test_dataset(action_data, action_label)
    train_dataset, test_dataset = util.split_train_test_dataset()
    train_with_simple_sequential_model(train_dataset, test_dataset)
    