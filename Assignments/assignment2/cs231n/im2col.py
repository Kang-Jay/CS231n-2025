from builtins import range
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ~~START DELETE~~
def get_num_fields(x_shape, field_height, field_width, padding, stride):
    """
    Helper function to get the number of receptive fields in the horizontal
    and vertical directions.

    Inputs:
    - x_shape: A 4-tuple (N, C, H, W) giving the shape of the input
    - field_height: Integer; height of each receptive field
    - field_width: Intger; width of each receptive field
    - padding: The amount of padding that will be applied to the input
    - stride: The gap (horizontal and vertical) between adjacent receptive fields

    Returns: A tuple HH, WW indicating that there are HH receptive fields along
    the vertical and horzontal directions respectively of each input.
    """
    N, C, H, W = x_shape
    if (W + 2 * padding - field_width) % stride != 0:
        raise ValueError("Invalid params for im2col; width does not work")
    if (H + 2 * padding - field_height) % stride != 0:
        raise ValueError("Invalid params for im2col; height does not work")

    # There are WW and HH receptive fields in the x and y direction respectively
    WW = (W + 2 * padding - field_width) // stride + 1
    HH = (H + 2 * padding - field_height) // stride + 1

    return HH, WW


def field_coords(H, W, field_height, field_width, padding, stride):
    """
    Iterator to yield coordinates of receptive fields in the correct order.
    In particular, you should iterate over the input from left to right and
    then from top to bottom, like you were reading a book (where the array
    index (0, 0) is at the upper left hand corner).

    We use the yield keyword to implement this as a generator. Your code
    should look something like this:

    [loop over y0]:
      y1 = y0 + field_height
      [loop over x0]:
        x1 = x0 + field_width
        yield (y0, y1, x0, x1)

    We can then use field_coords to easily iterate over receptive fields
    coordinates like this:

    for y0, y1, x0, x1 in field_coords(*args):
      # Do something with the coordinates

    Inputs:
    - H: The height of each input
    - W: The width of each input
    - field_height: Integer; height of each receptive field
    - field_width: Intger; width of each receptive field
    - padding: The amount of padding to apply to the input
    - stride: The distance between adjacent receptive fields

    Yields:
    Tuples (y0, y1, x0, x1) giving coordinates of receptive fields.
    If x were an array of shape (H, W) and we wanted to iterate over receptive
    fields of x, then each x[y0:y1, x0:x1] would be a receptive field of x.
    """
    if (W + 2 * padding - field_width) % stride != 0:
        raise ValueError("Invalid params for field_coords; width does not work")
    if (H + 2 * padding - field_height) % stride != 0:
        raise ValueError("Invalid params for field_coords; height does not work")
    yy = 0
    while stride * yy + field_height <= H + 2 * padding:
        y0 = yy * stride
        y1 = yy * stride + field_height
        xx = 0
        while stride * xx + field_width <= W + 2 * padding:
            x0 = xx * stride
            x1 = xx * stride + field_width
            yield (y0, y1, x0, x1)
            xx += 1
        yy += 1


def im2col_naive(x, field_height=3, field_width=3, padding=1, stride=1):
    """
    Convert a 4D array of independent 3D inputs into a single 2D array where each
    column is a receptive field of one of the inputs.

    The input x has shape (N, C, H, W); this should be interpreted as N
    independent inputs, each with height H, width H, and C channels.

    A receptive field of a single input is a rectangular block that spans all
    channels and whose height and width are given by field_height and field_width
    respectively. We imagine sliding these receptive fields over each of the
    inputs, where the horizontal and vertical offset between each receptive field
    is given by the stride parameter.

    Before extracting receptive fields, we also zero-pad the top, bottom,
    left, and right of each input with zeros.

    We will use this to efficiently implement convolution and pooling layers.

    As a simple example, we run the following matrix through im2col with
    field_height=field_width=2, padding=1, and stride=1:

    [1 2]
    [3 4]

    We would first pad the matrix with 0, giving the following:

    [0 0 0 0]
    [0 1 2 0]
    [0 3 4 0]
    [0 0 0 0]

    We would then slide a 2x2 window over this padded array, and reshape each
    window to a column, producing the following:

    [0 0 0 0 1 2 0 3 4]
    [0 0 0 1 2 0 3 4 0]
    [0 1 2 0 3 4 0 0 0]
    [1 2 0 3 4 0 0 0 0]

    You should use the field_coords generator function from above to iterate over
    receptive fields. So that downstream reshape operations work properly, you
    should iterate first over receptive fields and then over inputs. In other
    words, you should have a loop that looks like this:

    next_col = 0
    for y0, y1, x0, x1 in field_coords(params):
      for i in xrange(num_inputs):
        cols[:, next_col] = [something]
        next_col += 1

    Inputs:
    - x: 4D array of shape (N, C, H, W). This should be interpreted as N
      independent inputs, each with height H, width W, and C channels.
    - field_height: Integer; height of each receptive field
    - field_width: Intger; width of each receptive field
    - padding: The number of pixels of zero-padding to apply to x before
      extracting receptive fields.
    - stride: The horizontal and vertical offsets between adjacent receptive
      fields.

    Returns:
    A 2D array where each column is a receptive field of one of the inputs.
    """
    # First figure out what the size of the output should be
    N, C, H, W = x.shape
    HH, WW = get_num_fields(x.shape, field_height, field_width, padding, stride)

    # Next we zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # Allocate space for the result. To figure out the shape, remember that each
    # column is a receptive field of one of the inputs, and that we need to store
    # all receptive fields for all inputs.
    cols = np.zeros((C * field_height * field_width, N * HH * WW), dtype=x.dtype)

    # Next we iterate over all inputs and copy receptive fields from x to cols.
    # We can probably make this more efficient with some fancy indexing.
    next_col = 0
    for y0, y1, x0, x1 in field_coords(
        H, W, field_height, field_width, padding, stride
    ):
        for i in range(N):
            cols[:, next_col] = x_padded[i, :, y0:y1, x0:x1].flatten()
            next_col += 1
    return cols


def col2im(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    Perform an operation which is not quite the inverse of im2col that we need
    to efficiently compute the backward pass of a convolutional layer.

    Specifically, cols is a matrix where each column holds the data for a
    receptive field, and x_shape is a tuple giving the shape of the original 4D
    block of data from which cols was formed. We want to reshape the data in cols
    back to the shape given by x_shape; where multiple receptive fields overlap,
    we want to sum the corresponding elements of x_shape.

    As a simple example, we know that im2col with filter_height=filter_width=2,
    padding=0 and stride=2 would produce the following transformation:

    [1 2 3]               [1 2 4 5]
    [4 5 6]  --im2col-->  [2 3 5 6]
    [7 8 9]               [4 5 7 8]
                          [5 6 8 9]

    In contrast, col2im with the same parameters would produce the following:

    [a b c d]               [ a      e+b      f ]
    [e f g h]  --col2im-->  [i+c  m+j+k+g+d  n+h]
    [i j k l]               [ k      o+l      p ]
    [m n o p]

    To easily implement this, you can reuse the field_coords generator function.
    Make sure you iterate over receptive fields and then inputs as in im2col.

    If padding is nonzero, then col2im should throw away the portion of the
    reshaped array that corresponds to padding; in other words, the output
    of col2im should have shape equal to x_shape.

    Inputs:
    - cols: Array where each column is a receptive field
    - x_shape: Tuple (N, C, H, W) giving shape to which cols will be reshaped
    - field_height: Integer; height of each receptive field
    - field_width: Intger; width of each receptive field
    - padding: The number of pixels of zero-padding to apply to x before
      extracting receptive fields.
    - stride: The horizontal and vertical offsets between adjacent receptive
      fields.
    """
    x = np.empty(x_shape, dtype=cols.dtype)
    N, C, H, W = x_shape
    HH, WW = get_num_fields(x_shape, field_height, field_width, padding, stride)

    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    next_col = 0
    for y0, y1, x0, x1 in field_coords(
        H, W, field_height, field_width, padding, stride
    ):
        for i in range(N):
            col = cols[:, next_col]
            x_padded[i, :, y0:y1, x0:x1] += col.reshape(C, field_height, field_width)
            next_col += 1

    # Undo the padding
    if padding > 0:
        x = x_padded[:, :, padding:-padding, padding:-padding]
    else:
        x = x_padded
    assert x.shape == x_shape, "Expected shape %r but got shape %r" % (x_shape, x.shape)
    return x


# ~~END DELETE~~
