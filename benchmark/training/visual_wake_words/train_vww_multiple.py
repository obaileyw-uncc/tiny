# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.
"""

import os
import datetime

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')

import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 50

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')

TRAIN_RUNS = 2

# stepped learning rate as used in the original benchmark training script
# modified for use as a callback to allow a contiguous training history
# def lr_schedule(epoch):
#     adj1 = EPOCHS * 0.4
#     adj2 = EPOCHS * 0.6
#     if epoch >= adj2:
#         lr = 0.00025
#     elif epoch >= adj1:
#         lr = 0.0005
#     else:
#         lr = 0.001
#     if epoch in [0, adj1, adj2]:
#         if epoch == 0:
#            print(f"Learning rate adjusts after {int(adj1)} and {int(adj2)} epochs")
#         print(f"Learning rate: {lr}")
#     return lr

# cosine learning rate schedule with warmup
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, learning_rate_min, total_steps,
        warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.learning_rate_min = learning_rate_min
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            self.learning_rate_min +
            0.5
            * (self.learning_rate_base - self.learning_rate_min)
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
    
    def get_config(self):
        config = {
            "learning_rate_base": self.learning_rate_base,
            "learning_rate_min": self.learning_rate_min,
            "total_steps": self.total_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def main(argv):
  for model_filters in [12, 16, 17]:
    print(f"\n======{model_filters} FIRST LAYER FILTERS======")
    for train_run in range(TRAIN_RUNS):
        print(f"----------TRAINING RUN {train_run + 1}----------")
        model = mobilenet_v1(model_filters)
        if train_run == 0:
            model.summary()

        batch_size = 50
        validation_split = 0.1

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=.1,
            horizontal_flip=True,
            validation_split=validation_split,
            rescale=1. / 255)
        train_generator = datagen.flow_from_directory(
            BASE_DIR,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            subset='training',
            color_mode='rgb')
        val_generator = datagen.flow_from_directory(
            BASE_DIR,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            subset='validation',
            color_mode='rgb')
        print(train_generator.class_indices)

        lr_scheduler = None
        lr_scheduler = WarmUpCosine(learning_rate_base=0.001,
                                    learning_rate_min=0,
                                    total_steps=(EPOCHS * len(train_generator)),
                                    warmup_learning_rate=0.0,
                                    warmup_steps=1500)
        # verify learning rate schedule
        steps_plot = range(EPOCHS*len(train_generator))
        epochs_plot = [i / len(train_generator) for i in steps_plot]
        plt.clf()
        plt.plot(epochs_plot, lr_scheduler(tf.convert_to_tensor(steps_plot)))
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        plt.title('Cosine decay learning rate')
        plt.savefig('lr_schedule.png')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        batch_size=BATCH_SIZE)

        # Save model HDF5
        dt = datetime.datetime.today()
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        timestamp = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
        model_name = f"vww_96_{model_filters}_{train_run}_{timestamp}"
        model.save(f"trained_models/{model_name}.h5")
        print(f"Model saved to trained_models/{model_name}.h5\n")
        plt.clf() # clear figure in case the previous figure was not plotted
        plt.plot(np.array(range(EPOCHS)), history.history['loss'])
        plt.plot(np.array(range(EPOCHS)), history.history['val_loss'])
        plt.legend(labels=['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f"trained_models/{model_name}_train_val_loss.png")
        plt.clf()
        plt.plot(np.array(range(EPOCHS)), history.history['accuracy'])
        plt.plot(np.array(range(EPOCHS)), history.history['val_accuracy'])
        plt.legend(labels=['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(f"trained_models/{model_name}_train_val_acc.png")


# def train_epochs(model, train_generator, val_generator, epoch_count,
#                  learning_rate):
#   model.compile(
#       optimizer=tf.keras.optimizers.Adam(learning_rate),
#       loss='categorical_crossentropy',
#       metrics=['accuracy'])
#   history_fine = model.fit(
#       train_generator,
#       steps_per_epoch=len(train_generator),
#       epochs=epoch_count,
#       validation_data=val_generator,
#       validation_steps=len(val_generator),
#       batch_size=BATCH_SIZE)
#   return model


if __name__ == '__main__':
  app.run(main)
