from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, LayerNormalization, Dropout, Lambda, SpatialDropout3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate, Subtract


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 7, padding="same")(input)
    x = LayerNormalization(epsilon=1e-6)(x)   #Not in the original network.
    # x = SpatialDropout3D(0.1)(x)
    x = Activation("gelu")(x)

    x = Conv2D(num_filters, 7, padding="same")(x)
    x = LayerNormalization(epsilon=1e-6)(x)  #Not in the original network
    # x = SpatialDropout3D(0.1)(x)
    x = Activation("gelu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = Conv2D(num_filters, 2, strides=(2,2), activation='gelu')(x)#MaxPooling2D((2, 2),strides=2)(x)
    return x, p   
#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters, stride = 2):
    x = Conv2DTranspose(num_filters, (7, 7), strides=stride, padding="same", activation='gelu')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 32)
    s3, p3 = encoder_block(p2, 32)
    #s4, p4 = encoder_block(p3, 32)

    b1 = conv_block(p3, 32) #Bridge

    d1 = decoder_block(b1, s3, 32)
    d2 = decoder_block(d1, s2, 32)
    d3 = decoder_block(d2, s1, 32)
    #d4 = decoder_block(d3, s1, 32)

    outputs = Conv2D(n_classes, (1, 1), padding="same")(d3)  #Change the activation based on n_classes
    
    model = Model(inputs, outputs, name="U-Net")
    return model
