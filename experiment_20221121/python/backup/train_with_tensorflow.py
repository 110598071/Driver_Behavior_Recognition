import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import util

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
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, len(train_dataset[0][0])), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=(None, train_dataset[1][0]), name='tf_y')

    h1 = tf.layers.dense(inputs=tf_x, units=25, activation=tf.tanh, name='layer1')
    h2 = tf.layers.dense(inputs=h1, units=10, activation=tf.tanh, name='layer2')
    h3 = tf.layers.dense(inputs=h2, units=5, activation=tf.tanh, name='layer3')

if __name__ == '__main__':
    # train_dataset, val_dataset, test_dataset = util.split_train_val_test_dataset(action_data, action_label)
    train_dataset, test_dataset = util.split_train_test_dataset()
    train_with_simple_sequential_model(train_dataset, test_dataset)