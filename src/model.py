"""ResScene-CNN v2: a squeeze-and-excitation residual CNN for 150x150 scene images,
plus a from-scratch Grad-CAM implementation.

Architecture code is taken verbatim from the training notebook (Section 3).
Grad-CAM is the notebook's Section 6(b) function, refactored to take the model as an
argument. Total parameters: 1,309,794 (1,306,914 trainable, 2,880 non-trainable).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def se_module(x, filters, block_id, r=8):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018), reduction r=8."""
    s = layers.GlobalAveragePooling2D(name=f"block{block_id}_se_squeeze")(x)
    s = layers.Dense(filters // r, activation="relu", name=f"block{block_id}_se_reduce")(s)
    s = layers.Dense(filters, activation="sigmoid", name=f"block{block_id}_se_excite")(s)
    s = layers.Reshape((1, 1, filters), name=f"block{block_id}_se_reshape")(s)
    return layers.Multiply(name=f"block{block_id}_se_scale")([x, s])


def residual_block(x, filters, block_id, spatial_dropout=None):
    """Conv-BN-ReLU-Conv-BN-SE wrapped in an identity/projection skip, then MaxPool."""
    shortcut = x

    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"block{block_id}_conv1")(x)
    y = layers.BatchNormalization(name=f"block{block_id}_bn1")(y)
    y = layers.Activation("relu", name=f"block{block_id}_relu1")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"block{block_id}_conv2")(y)
    y = layers.BatchNormalization(name=f"block{block_id}_bn2")(y)
    y = se_module(y, filters, block_id)  # channel attention on the residual branch

    # Projection shortcut (1x1 conv) whenever the channel count changes
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                 name=f"block{block_id}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"block{block_id}_proj_bn")(shortcut)

    out_name = "gradcam_target" if block_id == 4 else f"block{block_id}_out"
    y = layers.Add(name=f"block{block_id}_add")([y, shortcut])
    y = layers.Activation("relu", name=out_name)(y)

    y = layers.MaxPooling2D(2, name=f"block{block_id}_pool")(y)
    if spatial_dropout:
        y = layers.SpatialDropout2D(spatial_dropout, name=f"block{block_id}_sdrop")(y)
    return y


def build_resscene_cnn(input_shape=(150, 150, 3), num_classes=6):
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = residual_block(inputs, 32, block_id=1)
    x = residual_block(x, 64, block_id=2)
    x = residual_block(x, 128, block_id=3, spatial_dropout=0.3)
    x = residual_block(x, 256, block_id=4, spatial_dropout=0.3)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc_256")(x)
    x = layers.Dropout(0.5, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="ResScene_CNN_v2")


def grad_cam(model, img_float01, class_index, target_layer="gradcam_target"):
    """Grad-CAM (Selvaraju et al., 2017) from scratch with tf.GradientTape.

    img_float01: (H, W, 3) array in [0, 1]. Returns a heatmap in [0, 1] at the
    resolution of the target layer (resize it to the image size for overlays).
    """
    grad_model = keras.Model(inputs=model.input,
                             outputs=[model.get_layer(target_layer).output, model.output])
    x = tf.convert_to_tensor(np.asarray(img_float01)[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        class_score = preds[:, class_index]
    grads = tape.gradient(class_score, conv_out)          # dy^c / dA, shape (1, H, W, K)
    alpha = tf.reduce_mean(grads, axis=(1, 2))            # global-average-pooled gradients (1, K)
    cam = tf.einsum("bhwk,bk->bhw", conv_out, alpha)      # sum_k alpha_k * A^k
    cam = tf.nn.relu(cam)[0].numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


if __name__ == "__main__":
    m = build_resscene_cnn()
    print(f"Total parameters: {m.count_params():,}")
