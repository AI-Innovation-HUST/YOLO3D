import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Loss


from tensorflow.keras import backend as K

ARCH = MobileNetV2
BIN, OVERLAP = 6, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram']
BATCH_SIZE = 8
AUGMENTATION = False

# Define loss func
###### Training ##########
@tf.keras.saving.register_keras_serializable()
def orientation_loss(y_true, y_pred):
    # Find number of anchors
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    
    # Define the loss
    loss = -(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    loss = tf.reduce_sum(loss, axis=1)
    epsilon = 1e-5  ##small epsilon value to prevent division by zero.
    anchors = anchors + epsilon
    loss = loss / anchors
    loss = tf.reduce_mean(loss)
    loss = 2 - 2 * loss 

    return loss


def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)
# Construct the network
base_model = ARCH(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model = ARCH(weights=None, include_top=True, input_shape=input_shape)


# Add additional layers for orientation prediction
x = base_model.output
x = GlobalAveragePooling2D()(x)

dimension   = Dense(512)(x)
dimension   = LeakyReLU(alpha=0.1)(dimension)
dimension   = Dropout(0.2)(dimension)
dimension   = Dense(3)(dimension)                                               
dimension   = LeakyReLU(alpha=0.1, name='dimension')(dimension)

orientation = Dense(256)(x)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Dropout(0.2)(orientation)
orientation = Dense(BIN*2)(orientation)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Reshape((BIN,-1))(orientation)                                    
orientation = Lambda(l2_normalize, name='orientation')(orientation)

confidence  = Dense(256)(x)
confidence  = LeakyReLU(alpha=0.1)(confidence)
confidence  = Dropout(0.2)(confidence)
confidence  = Dense(BIN, activation='softmax', name='confidence')(confidence)  

model_mobilenet = Model(inputs= base_model.input, outputs=[dimension, orientation, confidence])
minimizer = Adam(learning_rate=1e-5)
model_mobilenet.compile(
                optimizer=minimizer,
                loss={
                    'dimension': 'mean_squared_error', 
                    # 'dimension': 'mean_absolute_error', 
                    # 'dimension': 'mean_squared_logarithmic_error', 
                    'orientation': orientation_loss, 
                    'confidence': 'binary_crossentropy',
                    # 'confidence': 'categorical_crossentropy',
                    },
                loss_weights={'dimension': 5.0, 'orientation': 1.5, 'confidence': 0.5},
                metrics={
                    'dimension': 'mse', 
                    'orientation': 'mse', 
                    'confidence': 'accuracy',
                    })




