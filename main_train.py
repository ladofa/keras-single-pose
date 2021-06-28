import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from params import args

from dataset import dv, dt


model = keras.applications.MobileNetV3Small(
    input_shape=(args.input_height, args.input_width, 3),
    include_top=False)
input = layers.Input((args.input_height, args.input_width, 3), dtype=tf.float32)
feat = model(input)

#points_label
x = feat
x = layers.Dense(17, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Reshape((-1, 17))(x)
x = layers.Permute((2, 1))(x)
points_label = layers.Softmax()(x)

#offset
x = feat
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(34, use_bias=False)(x)
x = layers.BatchNormalization()(x)
offset = layers.Reshape([17, 2])(x)

model = keras.models.Model(inputs=[input], outputs=[points_label, offset])

def my_loss_fn(y_true, y_pred):
    print(y_true)
    valid = tf.cast(y_true[2], tf.float32)
    points_label_true = y_true[0]
    offset_true = y_true[1]

    label_loss = tf.losses.sparse_categorical_crossentropy(y_true[0], y_pred[0])
    offset_loss = tf.losses.mse(y_true[1], y_pred[1])
    loss = tf.reduce_mean((label_loss + offset_loss) * valid, axis=-1)
    return loss

optimizer = keras.optimizers.Adam()

def calc_loss(y_true, y_pred):
    valid = tf.cast(y_true[2], tf.float32)
    points_label_true = y_true[0]
    offset_true = y_true[1]

    label_loss = tf.losses.sparse_categorical_crossentropy(y_true[0], y_pred[0]) * valid
    offset_loss = tf.losses.mse(y_true[1], y_pred[1]) * valid
    label_loss = tf.reduce_mean(label_loss)
    offset_loss = tf.reduce_mean(offset_loss)
    return label_loss, offset_loss

for epoch in range(args.epochs):
    for train_data in dt:
        with tf.GradientTape() as tape:
            y_pred = model(train_data[0])
            y_true = train_data[1]
            label_loss, offset_loss = calc_loss(y_true, y_pred)
            loss = label_loss + offset_loss
            print('\r', epoch, ':', label_loss, offset_loss, end='')
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    total_label_loss = 0
    total_offset_loss = 0
    count = 0
    for val_data in dv:
        y_pred = model(train_data[0])
        y_true = train_data[1]
        label_loss, offset_loss = calc_loss(y_true, y_pred)
        total_label_loss += label_loss
        total_offset_loss += offset_loss
    print('label loss, offset loss = ')
    print('    ', total_label_loss / count, ',', total_offset_loss / count)
    model.save('saved_data')
        


