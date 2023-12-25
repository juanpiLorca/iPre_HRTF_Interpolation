import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tqdm import tqdm

from losses import SpectralLoss
from metrics import LSDMetric

from Model import get_models

NUM_TRAINING = "0_full"

# Metrics utils: ------------------------------------------------------------------------------------
def plot_metrics(train_losses, losses_name, train_metric_l, l_metric_name, train_metric_r, 
                 r_metric_name, title):
    plt.title(title)
    plt.plot(train_losses,color='blue',label=losses_name)
    plt.plot(train_metric_l,color='green',label=l_metric_name)
    plt.plot(train_metric_r,color="red",label=r_metric_name)
    plt.show()


# Dataset loading: ---------------------------------------------------------------------------------

# Hyperparameters: 
BATCH_SIZE = 64 
num_coordinates = 3
elev_azimuth = 2

path_data = "dataset/shuffle to train/training_data_cartesian_full.npy"
dataset = np.load(path_data)
print(dataset.shape)
hrir_data = dataset[:, :, :-(num_coordinates+elev_azimuth)]
spatial_info = dataset[:, :, -(num_coordinates+elev_azimuth):]
spatial_info = spatial_info[:, :, :-elev_azimuth]
training_dataset = tf.concat([hrir_data, spatial_info], axis=-1)

# Training:
dataset_to_train = tf.data.Dataset.from_tensor_slices(training_dataset)
dataset_to_train = dataset_to_train.batch(BATCH_SIZE)


# Model: --------------------------------------------------------------------------------------------------
CVAE = get_models()
CVAE.summary()

# Training: -----------------------------------------------------------------------------------------------
# Hyperparameters: 
# could use learning rate schedules: 
# alpha_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, 
#                                                                       values=values2)
# alpha_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=init_alpha, 
#                                                                 decay_rate=decay_rate, 
#                                                                 decay_steps=decay_step)
num_epochs = 500
alpha_schedule = 1e-4

# Optimizer: 
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_schedule) 

# Loss: 
loss_object = SpectralLoss()
# Metric: 
loss_metric_left = tf.keras.metrics.MeanAbsoluteError()
loss_metric_right = tf.keras.metrics.MeanAbsoluteError()


# Custom training: 
@tf.function
def apply_gradient(optimizer, loss_object, model, x_true_left, x_true_right, x): 
    """
    applies the gradients to the trainable model weights
    """

    with tf.GradientTape() as tape: 
        # Compute the reconstruction
        x_left, x_right = model(x)
        # Compute loss
        loss_value = loss_object(y_true=[x_true_left[tf.newaxis, :], 
                                         x_true_right[tf.newaxis, :]], 
                                 y_pred=[x_left[tf.newaxis, :], 
                                         x_right[tf.newaxis, :]])
    
    # compute and apply gradients from both losses:
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return [x_left, x_right], loss_value


def train_data_for_one_epoch(train_dataset, optimizer, model, loss_object, loss_metric_left, loss_metric_right, verbose=True):
    """
    Computes the loss then updates the weights and metrics for one epoch.
    """

    losses = []
    pbar = tqdm(total=len(list(enumerate(train_dataset))), position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    # Iterate through all batches of training data
    for step, x_batch_train in enumerate(train_dataset):
        x = x_batch_train
        x_true_left = x[:, 0, :-num_coordinates]
        x_true_right = x[:, 1, :-num_coordinates]
        # Calculate loss and update trainable variables using optimizer
        recon_batch, loss_value = apply_gradient(optimizer, loss_object, model, x_true_left, x_true_right, x)
        losses.append(loss_value)

        # compute metrics: 
        loss_metric_left.update_state(x_true_left, recon_batch[0])
        loss_metric_right.update_state(x_true_right, recon_batch[1])

        # Update progress
        if verbose:
            pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
            pbar.update()  
    return losses


# Training Loop: 
epochs_train_losses, epoch_train_metrics_left, epoch_train_metrics_right = [], [], []
for epoch in range(num_epochs):
    print('Start of epoch %d' % (epoch,))
    
    # Train:
    losses_train = train_data_for_one_epoch(train_dataset=dataset_to_train, 
                                            optimizer=optimizer, 
                                            loss_object=loss_object, 
                                            model=CVAE,
                                            loss_metric_left=loss_metric_left, 
                                            loss_metric_right=loss_metric_right)
    # Get results from training metrics
    train_acc_left = loss_metric_left.result()
    epoch_train_metrics_left.append(train_acc_left)
    train_acc_right = loss_metric_right.result()
    epoch_train_metrics_right.append(train_acc_right)

    #Calculate training and validation losses for current epoch
    losses_train_mean = np.mean(losses_train)
    epochs_train_losses.append(losses_train_mean)

    epoch_performance = "\nEpoch: {}, Train loss: {}, Train Accuracy Left: {}, Train Accuracy Right: {}"
    print(epoch_performance.format(epoch, float(losses_train_mean), float(train_acc_left), float(train_acc_right)))
    #Reset states of all metrics
    loss_metric_left.reset_states()
    loss_metric_right.reset_states()


plot_metrics(epochs_train_losses, "Loss", epoch_train_metrics_left, "Metrics Left", 
             epoch_train_metrics_right, "Metrics Right", "Losses")

# Saving wheights & training data: ---------------------------------------------------------------------------------------
weights = f"training data/weights/weights_cvae{NUM_TRAINING}_on_batch{BATCH_SIZE}_epochs{num_epochs}.h5"
CVAE.save_weights(weights)
print("Your model's weights were succesfully stored in the weights folder!")

loss_data = f"training data/losses per epoch/losses_cvae{NUM_TRAINING}_on_batch{BATCH_SIZE}_epochs{num_epochs}.npy"
epochs_train_losses = np.array(epochs_train_losses)
np.save(loss_data, epochs_train_losses)
