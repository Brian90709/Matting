import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def diff_x(input, r):
    assert input.ndim == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = paddle.concat([left, middle, right], axis=2)

    return output

def diff_y(input, r):
    assert input.ndim == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = paddle.concat([left, middle, right], axis=3)

    return output

class BoxFilter(nn.Layer):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.ndim == 4

        cumsum_x = paddle.cumsum(x, axis=2)
        diff_x_result = diff_x(cumsum_x, self.r)
        diff_y_result = paddle.cumsum(diff_x_result, axis=3)
        result = diff_y(diff_y_result, self.r)

        return result

class GuidedFilter(nn.Layer):
    def __init__(self, r, eps=1e-8):
        super().__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.shape
        n_y, c_y, h_y, w_y = y.shape

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        N = self.boxfilter(paddle.ones(shape=[1, 1, h_x, w_x], dtype='float32'))
        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b
    
class ConvGuidedFilter(nn.Layer):
    def __init__(self, radius=1, norm=nn.BatchNorm2D):
        super().__init__()

        # define box_filter
        self.box_filter = nn.Conv2D(1, 1, kernel_size=3, padding=radius, dilation=radius, bias_attr=False)

        # define conv_a
        self.conv_a = nn.Sequential(
            nn.Conv2D(2, 32, kernel_size=1, bias_attr=False),
            norm(32),
            nn.ReLU(),
            nn.Conv2D(32, 32, kernel_size=1, bias_attr=False),
            norm(32),
            nn.ReLU(),
            nn.Conv2D(32, 1, kernel_size=1, bias_attr=False)
        )

        # initialize weights of box_filter 
        self.box_filter.weight.set_value(paddle.full(self.box_filter.weight.shape, 1.0, dtype='float32'))


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.shape

        N = self.box_filter(paddle.full(shape=[1, 1, h_x, w_x], fill_value=1.0, dtype='float32'))
        # mean_x
        mean_x = self.box_filter(x) / N
        # mean_y
        mean_y = self.box_filter(y) / N
        # cov_xy
        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.box_filter(x * x) / N - mean_x * mean_x

        # A
        mean_A = self.conv_a(paddle.concat([cov_xy, var_x], axis=1))
        # b
        mean_b = mean_y - mean_A * mean_x

        return mean_A * x + mean_b