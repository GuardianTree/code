from keras.models import Model
import six
from keras.layers import Input, Concatenate, Conv2D
from keras import backend as K
from functools import partial
import model_utils
from capsnets_laseg.models.capsnets import CapsNetR3, CapsNetBasic



def SimpleUNet(input_shape):
    """
    Builds a simple U-Net with batch normalization.
    Args:
        include_top: whether to include the final sigmoid layer
        input_shape: (x, y, n_channels)
        activation:
        padding: layer padding (default: 'same')
    """
    #para
    include_top = False
    input_layer = None
    #input_shape = (None, None, None)
    starting_filters = 32
    depth = 4
    n_convs = 2
    activation = 'relu'
    padding = 'same'
    # initializing some reusable components
    context_mod = partial(model_utils.context_module_2D, n_convs = n_convs, activation = activation)
    localization_mod = partial(model_utils.localization_module_2D, n_convs = n_convs, activation = activation, \
                               transposed_conv = True)
    filter_list = [starting_filters*(2**level) for level in range(0, depth)]
    pool_size = (2,2)
    max_pools = depth - 1
    # start of building the model
    if input_layer is None:
        input_layer = Input(shape = input_shape, name = 'x')
    skip_layers = []
    level = 0
    # context pathway (downsampling) [level 0 to (depth - 1)]
    while level < max_pools:
        if level == 0:
            skip, pool = context_mod(input_layer, filter_list[level], pool_size = pool_size)
        elif level > 0:
            skip, pool = context_mod(pool, filter_list[level], pool_size = pool_size)
        skip_layers.append(skip)
        level += 1
    convs_bottom = context_mod(pool, filter_list[level], pool_size = None) # No downsampling;  level at (depth) after the loop
    convs_bottom = context_mod(convs_bottom, filter_list[level], pool_size = None) # happens twice
    # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
    while level > 0: # (** level = depth - 1 at the start of the loop)
        current_depth = level - 1
        if level == max_pools:
            upsamp = localization_mod(convs_bottom, skip_layers[current_depth], filter_list[current_depth],\
                                           upsampling_size = pool_size)
        elif not level == max_pools:
            upsamp = localization_mod(upsamp, skip_layers[current_depth], filter_list[current_depth],\
                                           upsampling_size = pool_size)
        level -= 1
    conv_transition = Conv2D(starting_filters, (1, 1), activation = activation)(upsamp)
    # return feature maps
    if not include_top:
        extractor = Model(inputs = [input_layer], outputs = [conv_transition])
        return extractor
    # return the segmentation
    elif include_top:
        # inferring the number of classes
        n_class = input_shape[-1]
        # setting activation function based on the number of classes
        if n_class > 1: # multiclass
            conv_seg = Conv2D(n_class, (1,1), activation = 'softmax')(conv_transition)
        elif n_class == 1:# binary
            conv_seg = Conv2D(1, (1,1), activation = 'sigmoid')(conv_transition)
        unet = Model(inputs = [input_layer], outputs = [conv_seg])
        return unet


class U_CapsNet(object):
    """
    The U-CapsNet architecture is made up of a U-Net feature extractor for a capsule network.
    Attributes:
        input_shape: sequence representing the input shape; (x, y, n_channels)
        n_class: the number of classes including the background class
        decoder: whether or not you want to include a reconstruction decoder in the architecture
    """
    def __init__(self, input_shape, n_class=2, decoder = True,):
        self.input_shape = input_shape
        self.n_class = n_class
        self.decoder = decoder

    def build_model(self, model_layer = None, capsnet_type = 'r3', upsamp_type = 'deconv'):
        """
        Builds the feature extractor + SegCaps network;
            Returns a keras.models.Model instance
        Args:
            model_layer: feature extractor
                * None: defaults to the AdaptiveUNet
                * 'simple': defaults to the basic U-Net
            capsnet_type: type of capsule network
                * 'r3':
                * 'basic':
            upsamp_type (str): one of ['deconv', 'subpix'] that represents the type of upsampling. Defaults to 'deconv'
        Returns:
            train_model: model for training
            eval_model: model for evaluation/inference
        """

        x = Input(self.input_shape, name = 'x')
        # initializing the U-Net feature extractor
        if model_layer is None:
            adap = AdaptiveUNet(2, self.input_shape, n_classes = self.n_class - 1, max_pools = 6, starting_filters = 5)
            model = adap.build_model(include_top = False, input_layer = x)
            # tensor_inp = model.output
        elif model_layer == "simple":
            model = SimpleUNet(include_top = False, input_layer = x, input_shape = self.input_shape)
            # tensor_inp = simp_u.output
        else:
            model = model_layer
        # intializing the Capsule Network
        if capsnet_type.lower() == 'r3':
            train_model, eval_model = CapsNetR3(self.input_shape, n_class = 2, decoder = self.decoder, add_noise = False, \
                                      input_layer = model, upsamp_type = upsamp_type)
        elif capsnet_type.lower() == 'basic':
            train_model, eval_model = CapsNetBasic(self.input_shape, n_class = 2, decoder = self.decoder, add_noise = False, \
                                      input_layer = model)
        return train_model, eval_model
