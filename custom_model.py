import numpy as np
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalMaxPooling1D, Dense
from tensorflow.keras import backend as K


def toy_model_io():
    input_src = tf.keras.layers.Input(shape=(128,), name="input_src_text")
    input_tgt = tf.keras.layers.Input(shape=(128,), name="input_tgt_text")
    x = tf.keras.layers.concatenate([input_src, input_tgt])
    x = Dense(2048, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    output = tf.keras.layers.Dense(2, activation="softmax")(x)

    return input_src, input_tgt, output

def toy_model():
    input_src, input_tgt, output = toy_model_io()
    model = tf.keras.Model([input_src, input_tgt], output)
    return model


class CustomToyModel(tf.keras.Model):

    def train_step(self, example):
        x, y = example
        with tf.GradientTape() as tape:
            output = self(x, training=True)
            loss = self.compiled_loss(y, output,
                                      sample_weight=None,
                                      regularization_losses=self.losses)

        variables = self.trainable_variables
        gradient = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradient, variables))

        self.compiled_metrics.update_state(y, output)

        return {m.name: m.result() for m in self.metrics}


"""Low level train step"""
def train_step(my_model, example, optimizer, loss_fn):

    with tf.GradientTape() as tape:
        output = my_model(example[0], training=True)
        loss = loss_fn(example[1], output)

    variables = my_model.trainable_variables
    gradient = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradient, variables))

    return loss

def train_and_checkpoint_per_epoch(model, ckpt, manager, data_iterator,
                                   optimizer=None,
                                   loss_fn=None,
                                   epochs=10):

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restoring from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing training from scratch")

    for epoch in range(epochs):
        print("\nTraining epoch: {}".format(epoch + 1))
        for example in data_iterator:
            #  print(example)
            loss_value = train_step(model, example, optimizer, loss_fn)

        #  ckpt.step.assign_add(1)
        save_path = manager.save()
        print("\tSaved checkpoint for epoch {}: {}".format(epoch + 1, save_path))
        print("\tLoss at final step {:1.2f}".format(loss_value.numpy()))


def create_dummy_data(num_examples=1000, batch_size=32):
    input_src_text = tf.random.uniform((num_examples, 128))
    input_tgt_text = tf.random.uniform((num_examples, 128))
    labels = np.random.randint(2, size=(num_examples,2))
    # train_data = tf.data.Dataset.from_tensor_slices(dict(
    #                                                  x = {"input_src_text": input_src_text,
    #                                                       "input_tgt_text": input_tgt_text},
    #                                                  y = labels)).batch(5)
    train_data = tf.data.Dataset.from_tensor_slices(({"input_src_text": input_src_text,
                                                      "input_tgt_text": input_tgt_text},
                                                      labels)).batch(batch_size)
    return train_data


def test_train():

    # training settings
    model_dir = "/linguistics/ethan/DL_Prototype/models/example_ckpt"

    data = create_dummy_data(num_examples=10000, batch_size=32)
    with tf.device("/gpu:7"):
        my_model = toy_model()
    optimizer = tf.keras.optimizers.Adam(0.1)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    data_iterator = iter(data)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               optimizer=optimizer,
                               net=my_model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

    # train
    train_and_checkpoint_per_epoch(my_model, ckpt, manager, data_iterator,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   epochs=10)


def test_custom_train():

    # prepare data
    data = create_dummy_data(num_examples=10000, batch_size=32)
    epochs = 10
    # Build a model with customized train step, train
    with tf.device("/gpu:7"):
        input_src, input_tgt, output = toy_model_io()
        custom_model = CustomToyModel([input_src, input_tgt], output)
        custom_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                             loss="categorical_crossentropy",
                             metrics=["accuracy"])
        custom_model.fit(data, epochs=epochs)


if __name__ == "__main__":

    # test_train()
    test_custom_train()
