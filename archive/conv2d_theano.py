def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.
    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "channels_last" or "channels_first".
            Whether to use Theano or TensorFlow data format
        in inputs/kernels/outputs.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ', data_format)

    if hasattr(x, '_keras_shape'):
        image_shape = _preprocess_conv2d_image_shape(int_shape(x), data_format)
    else:
        image_shape = None
    if hasattr(kernel, '_keras_shape'):
        kernel_shape = kernel._keras_shape
    else:
        # Will only work if `kernel` is a shared variable.
        kernel_shape = kernel.eval().shape
    kernel_shape = _preprocess_conv2d_filter_shape(kernel_shape, data_format)

    x = _preprocess_conv2d_input(x, data_format)
    kernel = _preprocess_conv2d_kernel(kernel, data_format)
    th_padding = _preprocess_padding(padding)

    conv_out = T.nnet.conv2d(x, kernel,
                             border_mode=th_padding,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=kernel_shape,
                             filter_dilation=dilation_rate)
    conv_out = _postprocess_conv2d_output(conv_out, x, padding,
                                          kernel_shape, strides, data_format)
return conv_out