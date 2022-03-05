from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten, Dense, UpSampling2D, UpSampling3D
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf


class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling
        for channel-first images and relative bounding box coordinates

        # Constructor parameters
            pooled_height, pooled_width (int) --
              specify height and width of layer outputs

        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]

        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    """

    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

        super(ROIPoolingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output

            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """

        def curried_pool_rois(x):
            return ROIPoolingLayer._pool_rois(x[0], x[1],
                                              self.pooled_height,
                                              self.pooled_width)

        pooled_areas = tf.map_fn(curried_pool_rois, x, fn_output_signature=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """

        def curried_pool_roi(roi):
            return ROIPoolingLayer._pool_roi(feature_map, roi,
                                             pooled_height, pooled_width)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, fn_output_signature=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        # Compute the region of interest
        feature_map_height = int(feature_map.shape[0])
        feature_map_width = int(feature_map.shape[1])
        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width * roi[1], 'int32')
        h_end = tf.cast(feature_map_height * roi[2], 'int32')
        w_end = tf.cast(feature_map_width * roi[3], 'int32')
        region = feature_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width = w_end - w_start
        h_step = tf.cast(region_height / pooled_height, 'int32')
        w_step = tf.cast(region_width / pooled_width, 'int32')

        areas = [[(
            i * h_step,
            j * w_step,
            (i + 1) * h_step if i + 1 < pooled_height else region_height,
            (j + 1) * w_step if j + 1 < pooled_width else region_width
        )
            for j in range(pooled_width)]
            for i in range(pooled_height)]

        # take the maximum of each area and stack the result
        def pool_area(x):
            return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0, 1])

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features

    def get_config(self):
        config = {'pool_height': self.pooled_height,
                  'pool_width': self.pooled_width}
        base_config = super(ROIPoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def RPN2ROI(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    pass


# Junk
# RoI Pooling Layer
class RoiPoolingConv(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'channels_first':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)
        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            x = tf.cast(x, 'int32')
            y = tf.cast(y, 'int32')
            w = tf.cast(w, 'int32')
            h = tf.cast(h, 'int32')

            rs = tf.image.resize(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Mask R-CNN
def Mask(BaseLayer):
    Out = UpSampling3D(size=(1, 2, 2))(BaseLayer)
    Out = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Out)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)
    Out = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Out)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)
    Out = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Out)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)
    Out = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Out)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)
    Out = UpSampling3D(size=(1, 2, 2))(Out)
    Out = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Out)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)
    Out = Conv2D(filters=80, kernel_size=(3, 3), padding="same", activation="relu")(Out)
    return Out


# Auto Encoder
def SegNet(BaseLayer, ClassesCount):
    Block6_Conv2D = UpSampling2D(size=(2, 2))(BaseLayer)
    Block6_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block6_Conv2D)
    Block6_Conv2D = Dropout(0.1)(Block6_Conv2D)
    Block6_Conv2D = BatchNormalization()(Block6_Conv2D)
    Block6_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block6_Conv2D)
    Block6_Conv2D = Dropout(0.1)(Block6_Conv2D)
    Block6_Conv2D = BatchNormalization()(Block6_Conv2D)
    Block6_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block6_Conv2D)

    Block7_Conv2D = UpSampling2D(size=(2, 2))(Block6_Conv2D)
    Block7_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block7_Conv2D)
    Block7_Conv2D = Dropout(0.1)(Block7_Conv2D)
    Block7_Conv2D = BatchNormalization()(Block7_Conv2D)
    Block7_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block7_Conv2D)
    Block7_Conv2D = Dropout(0.1)(Block7_Conv2D)
    Block7_Conv2D = BatchNormalization()(Block7_Conv2D)
    Block7_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block7_Conv2D)

    Block8_Conv2D = UpSampling2D(size=(2, 2))(Block7_Conv2D)
    Block8_Conv2D = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Block8_Conv2D)
    Block8_Conv2D = Dropout(0.1)(Block8_Conv2D)
    Block8_Conv2D = BatchNormalization()(Block8_Conv2D)
    Block8_Conv2D = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Block8_Conv2D)
    Block8_Conv2D = Dropout(0.1)(Block8_Conv2D)
    Block8_Conv2D = BatchNormalization()(Block8_Conv2D)
    Block8_Conv2D = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Block8_Conv2D)

    Block9_Conv2D = UpSampling2D(size=(2, 2))(Block8_Conv2D)
    Block9_Conv2D = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(Block9_Conv2D)
    Block9_Conv2D = Dropout(0.1)(Block9_Conv2D)
    Block9_Conv2D = BatchNormalization()(Block9_Conv2D)
    Block9_Conv2D = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(Block9_Conv2D)
    Block9_Conv2D = Dropout(0.1)(Block9_Conv2D)
    Block9_Conv2D = BatchNormalization()(Block9_Conv2D)

    Block10_Conv2D = UpSampling2D(size=(2, 2))(Block9_Conv2D)
    Block10_Conv2D = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(Block10_Conv2D)
    Block10_Conv2D = Dropout(0.1)(Block10_Conv2D)
    Block10_Conv2D = BatchNormalization()(Block10_Conv2D)
    Block10_Conv2D = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(Block10_Conv2D)
    Block10_Conv2D = Dropout(0.1)(Block10_Conv2D)
    Block10_Conv2D = BatchNormalization()(Block10_Conv2D)

    Out = Conv2D(filters=ClassesCount, kernel_size=(1, 1), padding="valid", activation="relu")(Block10_Conv2D)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)

    return Out


