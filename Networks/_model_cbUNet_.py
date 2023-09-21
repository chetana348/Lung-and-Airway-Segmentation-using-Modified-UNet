from keras_unet import TF
if TF:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        MaxPooling2D,
        Dropout,
        SpatialDropout2D,
        UpSampling2D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
    )
else:
    from keras.models import Model
    from keras.layers import (
        BatchNormalization,
        Conv2D,
        Conv2DTranspose,
        MaxPooling2D,
        Dropout,
        SpatialDropout2D,
        UpSampling2D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
    )



# Function for upsampling using convolution transpose
def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

# Function for simple upsampling
def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

# Function to create an attention gate
def attention_gate(input_1, input_2, n_intermediate_filters):
    # This function implements an attention gate as proposed by Oktay et al. in their Attention U-net.
    # It compresses both input tensors to n_intermediate_filters filters before processing.
    # For details, refer to: https://arxiv.org/abs/1804.03999
    input_1_conv = Conv2D(n_intermediate_filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(input_1)
    input_2_conv = Conv2D(n_intermediate_filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(input_2)
    
    f = Activation("relu")(add([input_1_conv, input_2_conv]))
    g = Conv2D(filters=1, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(f)
    h = Activation("sigmoid")(g)
    
    return multiply([input_1, h])

# Function to perform concatenation of upsampled convolutional layer with attention-gated skip-connection
def attention_concat(conv_layer_below, skip_connection):
    below_filters = conv_layer_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_layer_below, below_filters)
    return concatenate([conv_layer_below, attention_across])

# Function to create a convolutional block
def conv2d_block(inputs, use_batch_norm=True, dropout=0.3, dropout_type="spatial", filters=16,
                 kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same"):
    # This function defines a convolutional block with optional batch normalization and dropout.
    # It's a fundamental building block of the UNet architecture.

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}")

    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding, use_bias=not use_batch_norm)(inputs)
    
    if use_batch_norm:
        c = BatchNormalization()(c)
    
    if dropout > 0.0:
        c = DO(dropout)(c)
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding, use_bias=not use_batch_norm)(c)
    
    if use_batch_norm:
        c = BatchNormalization()(c)
    
    return c

# Function to create a custom UNet model
def cbUNet(input_shape, num_classes=1, activation="relu", use_batch_norm=True, upsample_mode="deconv",
                dropout=0.3, dropout_change_per_layer=0.0, dropout_type="spatial", use_dropout_on_upsampling=False,
                use_attention=False, filters=16, num_layers=4, output_activation="sigmoid"):
    # This function defines a customizable UNet architecture based on user-specified parameters.


    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    
    # Build the encoder part of the UNet
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, dropout_type=dropout_type, activation=activation)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, dropout_type=dropout_type, activation=activation)
    
    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    # Build the decoder part of the UNet
    for conv in reversed(down_layers):
        filters //= 2
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        
        if use_attention:
            x = attention_concat(conv_layer_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, dropout_type=dropout_type, activation=activation)

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
