from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Input,
    MaxPooling2D,
    Permute,
    Reshape,
    UpSampling2D,
)


def track_net(
    input_n_images: int = 1,
    input_height: int = 360,
    input_width: int = 640,
    n_classes: int = 256,
    return_summary: bool = True,
):
    """
    Creates TrackNet for tracking small, fast-moving objects such as a tennis ball.

    Parameters:
        input_n_images (int): number of images as input
        input_height (int): height of the input image
        input_width (int): width of the input image
        n_classes (int): number of classes in the dataset
        return_summary (bool): whether to return the summary of the model or not

    Returns:
        None

    Adapted from:
    https://nol.cs.nctu.edu.tw:234/open-source/TrackNet/blob/master/Code_Python2/TrackNet_Three_Frames_Input/Models/TrackNet.py
    """
    imgs_input = Input(shape=(input_n_images * 3, input_height, input_width))

    # layer1
    x = Conv2D(
        64,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(imgs_input)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer2
    x = Conv2D(
        64,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    # layer4
    x = Conv2D(
        128,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer5
    x = Conv2D(
        128,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer6
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    # layer7
    x = Conv2D(
        256,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer8
    x = Conv2D(
        256,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer9
    x = Conv2D(
        256,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer10
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x)

    # layer11
    x = (
        Conv2D(
            512,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer12
    x = (
        Conv2D(
            512,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer13
    x = (
        Conv2D(
            512,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer14
    x = (UpSampling2D((2, 2), data_format="channels_first"))(x)

    # layer15
    x = (
        Conv2D(
            256,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer16
    x = (
        Conv2D(
            256,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer17
    x = (
        Conv2D(
            256,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer18
    x = (UpSampling2D((2, 2), data_format="channels_first"))(x)

    # layer19
    x = (
        Conv2D(
            128,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer20
    x = (
        Conv2D(
            128,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer21
    x = (UpSampling2D((2, 2), data_format="channels_first"))(x)

    # layer22
    x = (
        Conv2D(
            64,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer23
    x = (
        Conv2D(
            64,
            (3, 3),
            kernel_initializer="random_uniform",
            padding="same",
            data_format="channels_first",
        )
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    # layer24
    x = Conv2D(
        n_classes,
        (3, 3),
        kernel_initializer="random_uniform",
        padding="same",
        data_format="channels_first",
    )(x)
    x = (Activation("relu"))(x)
    x = (BatchNormalization())(x)

    o_shape = Model(imgs_input, x).output_shape
    print("TrackNet Layer24 output shape:", o_shape[1], o_shape[2], o_shape[3])
    # layer24 output shape: 256, 360, 640

    OutputHeight = o_shape[2]
    OutputWidth = o_shape[3]

    # reshape the size to (256, 360*640)
    x = (Reshape((-1, OutputHeight * OutputWidth)))(x)

    # change dimension order to (360*640, 256)
    x = (Permute((2, 1)))(x)

    # layer25
    gaussian_output = (Activation("softmax"))(x)

    model = Model(imgs_input, gaussian_output)
    model.outputWidth = OutputWidth
    model.outputHeight = OutputHeight

    # show model details
    if return_summary:
        model.summary()

    return model


# def track_net_NHWC(
#     input_n_images: int = 1,
#     input_height: int = 360,
#     input_width: int = 640,
#     n_classes: int = 256,
#     return_summary: bool = True,
# ):
#     """
#     Creates TrackNet for tracking small, fast-moving objects such as a tennis ball.

#     Parameters:
#         input_n_images (int): number of images as input
#         input_height (int): height of the input image
#         input_width (int): width of the input image
#         n_classes (int): number of classes in the dataset
#         return_summary (bool): whether to return the summary of the model or not

#     Returns:
#         None

#     Adapted from:
#     https://nol.cs.nctu.edu.tw:234/open-source/TrackNet/blob/master/Code_Python2/TrackNet_Three_Frames_Input/Models/TrackNet.py
#     """
#     imgs_input = Input(shape=(input_height, input_width, input_n_images * 3))

#     # layer1
#     x = Conv2D(
#         64,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(imgs_input)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer2
#     x = Conv2D(
#         64,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer3
#     x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last")(x)

#     # layer4
#     x = Conv2D(
#         128,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer5
#     x = Conv2D(
#         128,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer6
#     x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last")(x)

#     # layer7
#     x = Conv2D(
#         256,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer8
#     x = Conv2D(
#         256,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer9
#     x = Conv2D(
#         256,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer10
#     x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last")(x)

#     # layer11
#     x = (
#         Conv2D(
#             512,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer12
#     x = (
#         Conv2D(
#             512,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer13
#     x = (
#         Conv2D(
#             512,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer14
#     x = (UpSampling2D((2, 2), data_format="channels_last"))(x)

#     # layer15
#     x = (
#         Conv2D(
#             256,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer16
#     x = (
#         Conv2D(
#             256,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer17
#     x = (
#         Conv2D(
#             256,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer18
#     x = (UpSampling2D((2, 2), data_format="channels_last"))(x)

#     # layer19
#     x = (
#         Conv2D(
#             128,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer20
#     x = (
#         Conv2D(
#             128,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer21
#     x = (UpSampling2D((2, 2), data_format="channels_last"))(x)

#     # layer22
#     x = (
#         Conv2D(
#             64,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer23
#     x = (
#         Conv2D(
#             64,
#             (3, 3),
#             kernel_initializer="random_uniform",
#             padding="same",
#             data_format="channels_last",
#         )
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     # layer24
#     x = Conv2D(
#         n_classes,
#         (3, 3),
#         kernel_initializer="random_uniform",
#         padding="same",
#         data_format="channels_last",
#     )(x)
#     x = (Activation("relu"))(x)
#     x = (BatchNormalization())(x)

#     o_shape = Model(imgs_input, x).output_shape
#     print("TrackNet Layer24 output shape:", o_shape[1], o_shape[2], o_shape[3])
#     # layer24 output shape: 256, 360, 640

#     OutputHeight = o_shape[2]
#     OutputWidth = o_shape[3]

#     # reshape the size to (256, 360*640)
#     x = (Reshape((-1, OutputHeight * OutputWidth)))(x)

#     # change dimension order to (360*640, 256)
#     x = (Permute((2, 1)))(x)

#     # layer25
#     gaussian_output = (Activation("softmax"))(x)

#     model = Model(imgs_input, gaussian_output)
#     model.outputWidth = OutputWidth
#     model.outputHeight = OutputHeight

#     # show model details
#     if return_summary:
#         model.summary()

#     return model
