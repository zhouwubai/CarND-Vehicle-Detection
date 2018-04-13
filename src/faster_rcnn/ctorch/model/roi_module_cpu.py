from __future__ import division

import numpy as np
import six

from chainer import function
from chainer.utils import type_check


def _roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(np.floor(size * stride))
    end = int(np.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start


class PSROIPooling2D(function.Function):

    def __init__(self, out_c, out_h, out_w, spatial_scale, group_size):
        self.out_c, self.out_h, self.out_w = out_c, out_h, out_w
        self.spatial_scale = spatial_scale
        self.group_size = group_size

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type, roi_type, roi_index_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim == 4,
            roi_type.dtype == np.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 4,
            roi_index_type.dtype == np.int32,
            roi_index_type.ndim == 1,
            roi_type.shape[0] == roi_index_type.shape[0]
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_roi = bottom_rois.shape[0]
        top_data = np.empty(
            (n_roi, self.out_c, self.out_h, self.out_w), dtype=np.float32)

        for i_roi in six.moves.range(n_roi):
            y_min, x_min, y_max, x_max = bottom_rois[i_roi]
            batch_index = bottom_roi_indices[i_roi]
            y_min = round(y_min * self.spatial_scale)
            x_min = round(x_min * self.spatial_scale)
            y_max = round(y_max * self.spatial_scale)
            x_max = round(x_max * self.spatial_scale)
            roi_height = max(y_max - y_min, 0.1)
            roi_width = max(x_max - x_min, 0.1)

            stride_c = channels / self.out_c
            stride_h = roi_height / self.out_h
            stride_w = roi_width / self.out_w
            group_h = int(round(self.out_h / self.group_size))
            group_w = int(round(self.out_w / self.group_size))

            for out_h in six.moves.range(self.out_h):
                slice_h, len_h = _roi_pooling_slice(
                    out_h, stride_h, height, int(y_min))
                if slice_h.stop <= slice_h.start:
                    continue
                for out_w in six.moves.range(self.out_w):
                    slice_w, len_w = _roi_pooling_slice(
                        out_w, stride_w, width, int(x_min))
                    if slice_w.stop <= slice_w.start:
                        continue
                    for out_c in six.moves.range(self.out_c):
                        slice_c, len_c = _roi_pooling_slice(
                            out_c, stride_c, channels, 0)
                        roi_data = bottom_data[
                            batch_index, slice_c, slice_h, slice_w]\
                            .reshape((len_c, -1))
                        c = (out_h // group_h) * self.group_size \
                            + (out_w // group_w)
                        top_data[i_roi, out_c, out_h, out_w] = np.average(
                            roi_data[c])
        return top_data,

    def backward_cpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
        n_roi = bottom_rois.shape[0]
        bottom_diff = np.zeros(self._bottom_data_shape, np.float32)

        for i_roi in six.moves.range(n_roi):
            y_min, x_min, y_max, x_max = bottom_rois[i_roi]
            batch_index = bottom_roi_indices[i_roi]
            y_min = round(y_min * self.spatial_scale)
            x_min = round(x_min * self.spatial_scale)
            y_max = round(y_max * self.spatial_scale)
            x_max = round(x_max * self.spatial_scale)
            roi_height = max(y_max - y_min, 0.1)
            roi_width = max(x_max - x_min, 0.1)

            stride_c = channels / self.out_c
            stride_h = roi_height / self.out_h
            stride_w = roi_width / self.out_w
            group_h = int(round(self.out_h / self.group_size))
            group_w = int(round(self.out_w / self.group_size))

            for out_h in six.moves.range(self.out_h):
                slice_h, len_h = _roi_pooling_slice(
                    out_h, stride_h, height, int(y_min))
                if slice_h.stop <= slice_h.start:
                    continue
                for out_w in six.moves.range(self.out_w):
                    slice_w, len_w = _roi_pooling_slice(
                        out_w, stride_w, width, int(x_min))
                    if slice_w.stop <= slice_w.start:
                        continue
                    for out_c in six.moves.range(self.out_c):
                        diff_val = gy[0][i_roi, out_c, out_h, out_w]
                        diff_val = diff_val / len_h / len_w
                        start_c = int(np.floor(out_c * stride_c))
                        start_c = min(max(start_c, 0), channels)

                        c = (out_h // group_h) * self.group_size \
                            + (out_w // group_w) + start_c
                        bottom_diff[batch_index, c, slice_h, slice_w] \
                            += diff_val
        return bottom_diff, None, None


def psroi_pooling_2d(
        x, rois, roi_indices, out_c, out_h, out_w,
        spatial_scale, group_size
):
    """Position Sensitive Region of Interest (ROI) pooling function.
    This function computes position sensitive average of input spatial patch
    with the given region of interests. Each ROI is splitted into
    :math:`(group\_size, group\_size)` regions, and position sensitive values
    in each region is computed.
    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (array): Input roi. The shape is expected to
            be :math:`(R, 4)`, and each datum is set as below:
            (y_min, x_min, y_max, x_max). The dtype is :obj:`numpy.float32`.
        roi_indices (array): Input roi indices. The shape is expected to
            be :math:`(R, )`. The dtype is :obj:`numpy.int32`.
        out_c (int): Channels of output image after pooled.
        out_h (int): Height of output image after pooled.
        out_w (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.
        group_size (int): Position sensitive group size.
    Returns:
        ~chainer.Variable: Output variable.
    See the original paper proposing PSROIPooling:
    `R-FCN <https://arxiv.org/abs/1605.06409>`_.
    """
    return PSROIPooling2D(out_c, out_h, out_w, spatial_scale,
                          group_size)(x, rois, roi_indices)
