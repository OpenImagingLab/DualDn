import cv2
import numpy as np
from scipy.sparse import spdiags, vstack, hstack, block_diag, csc_matrix
from scipy.sparse.linalg import lsqr


def im2double(image):
    try:
        image = image.astype(np.float64) / 255.0
        return image
    except Exception as e:
        raise TypeError('Fail to convert to double')


def rgb2luminance(rgb, coeffs=[0.25, 0.5, 0.25]):

    # Check if the input is 3D (height x width x 3)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb should be a height x width x 3 array.")

    # Check if coeffs has exactly 3 elements
    if len(coeffs) != 3:
        raise ValueError("coeffs must be a 3-element vector.")

    # Warn if coefficients do not sum to 1
    if abs(1 - sum(coeffs)) > 1e-6:
        print(f"Warning: coeffs sum to {sum(coeffs)}, which is not 1.")
    
    luma = coeffs[0] * rgb[:, :, 0] + coeffs[1] * rgb[:, :, 1] + coeffs[2] * rgb[:, :, 2]
    return luma


def getDefaultAffineGridSize(input_image, output_image):
    input_height    = input_image.shape[0]
    input_width     = input_image.shape[1]
    input_channels  = input_image.shape[2]
    output_channels = output_image.shape[2]

    grid_size = np.round([input_height / 16, input_width / 16, 8, 
                    output_channels, input_channels + 1]).astype(int)

    return grid_size


def buildAffineSliceMatrix(input_image, edge_image, grid_size, i, j):
    num_grid_cells = np.prod(grid_size)
    image_height, image_width = input_image.shape[:2]
    num_pixels = image_width * image_height
    grid_height, grid_width, grid_depth, _, _ = grid_size

    pixel_x = np.arange(image_width)
    pixel_y = np.arange(image_height).reshape(-1, 1)

    # Convert to floating point bilateral grid coordinates
    bg_coord_x = (pixel_x + 0.5) * (grid_width - 1) / image_width
    bg_coord_y = (pixel_y + 0.5) * (grid_height - 1) / image_height
    bg_coord_z = edge_image * (grid_depth - 1)

    bg_idx_x0 = np.floor(bg_coord_x).astype(int)
    bg_idx_y0 = np.floor(bg_coord_y).astype(int)
    bg_idx_z0_im = np.floor(bg_coord_z).astype(int)
    bg_idx_x0_im = np.tile(bg_idx_x0, (image_height, 1))
    bg_idx_y0_im = np.tile(bg_idx_y0, (1, image_width))

    # Compute dx, dy, dz images
    dx = np.tile(bg_coord_x - bg_idx_x0, (image_height, 1))
    dy = np.tile(bg_coord_y - bg_idx_y0, (1, image_width))
    dz = bg_coord_z - bg_idx_z0_im

    # Compute the weights for each voxel
    weight_000 = ((1 - dx) * (1 - dy) * (1 - dz)).T.ravel()
    weight_100 = (dx * (1 - dy) * (1 - dz)).T.ravel()
    weight_010 = ((1 - dx) * dy * (1 - dz)).T.ravel()
    weight_110 = (dx * dy * (1 - dz)).T.ravel()
    weight_001 = ((1 - dx) * (1 - dy) * dz).T.ravel()
    weight_101 = (dx * (1 - dy) * dz).T.ravel()
    weight_011 = ((1 - dx) * dy * dz).T.ravel()
    weight_111 = (dx * dy * dz).T.ravel()

    # Create index arrays for the sparse matrix construction
    st_i = np.arange(1, 8 * num_pixels + 1)
    st_bg_xx = np.ravel(np.column_stack([bg_idx_x0_im.T.ravel() + 1, bg_idx_x0_im.T.ravel() + 2] * 4))
    st_bg_yy = np.ravel(np.column_stack(([bg_idx_y0_im.T.ravel() + 1] * 2 + [bg_idx_y0_im.T.ravel() + 2] * 2) * 2))
    st_bg_zz = np.ravel(np.column_stack([bg_idx_z0_im.T.ravel() + 1] * 4 + [bg_idx_z0_im.T.ravel() + 2] * 4))
    st_bg_uu = np.full(8 * num_pixels, i)
    st_bg_vv = np.full(8 * num_pixels, j)
    st_s = np.ones(8 * num_pixels)

    # Prune rows of the triplet vectors where grid indices go out of bounds
    indices = (st_bg_xx > 0) & (st_bg_xx <= grid_width) & (st_bg_yy > 0) & \
                (st_bg_yy <= grid_height) & (st_bg_zz > 0) & (st_bg_zz <= grid_depth)

    st_i = st_i[indices]
    st_bg_xx = st_bg_xx[indices]
    st_bg_yy = st_bg_yy[indices]
    st_bg_zz = st_bg_zz[indices]
    st_bg_uu = st_bg_uu[indices]
    st_bg_vv = st_bg_vv[indices]
    st_s = st_s[indices]
    st_j = np.ravel_multi_index((st_bg_yy-1, st_bg_xx-1, st_bg_zz-1, st_bg_uu-1, st_bg_vv-1), grid_size, order='F')

    st_m = 8 * num_pixels
    st_n = num_grid_cells

    # Create the sparse matrix for the slice
    st = csc_matrix((st_s, (st_i-1, st_j)), shape=(st_m, st_n))

    # Create weight matrix
    w_i = np.tile(np.arange(1, num_pixels + 1), (8, 1)).T.ravel()
    w_j = np.arange(1, 8 * num_pixels + 1)
    w_s = np.column_stack([weight_000, weight_100, weight_010, weight_110, weight_001, weight_101, weight_011, weight_111]).ravel()

    w_m = num_pixels
    w_n = 8 * num_pixels

    w = csc_matrix((w_s, (w_i-1, w_j-1)), shape=(w_m, w_n))

    return w, st


def buildApplyAffineModelMatrix(input_image, num_output_channels):
    num_pixels = input_image.shape[0] * input_image.shape[1]
    A = csc_matrix((0, 0))  # Initialize as an empty sparse matrix
    
    for k in range(input_image.shape[2]):
        plane = input_image[:, :, k]  # Flatten the 2D plane
        # Repeat each component num_output_channels times and build the sparse diagonal matrix
        sd = spdiags(np.tile(plane.T.ravel(), num_output_channels), 0, 
                     num_output_channels * num_pixels, num_output_channels * num_pixels)
        A = hstack([A, sd], format='csc')  # Horizontally concatenate the sparse matrices

    # Ones channel
    ones_diag = spdiags(np.ones(num_output_channels * num_pixels), 0,
                        num_output_channels * num_pixels, num_output_channels * num_pixels)
    
    A = hstack([A, ones_diag], format='csc')  # Add the ones diagonal as the final column

    return A


def buildDerivXMatrix(grid_size):
    # d/dx for every entry in the first slice except the last column
    m = grid_size[0] * (grid_size[1] - 1)
    n = grid_size[0] * grid_size[1]
    e = np.ones(m)
    d_dx = spdiags([-e, e], [0, grid_size[0]], m, n)
    
    A = csc_matrix((0, 0)) # Initialize empty sparse matrix
    
    for v in range(grid_size[4]): 
        for u in range(grid_size[3]):
            for k in range(grid_size[2]):
                A = block_diag([A, d_dx])  # Append block diagonal matrix

    return A


def buildDerivYMatrix(grid_size):
    # Derivatives down y.
    ny = grid_size[0]
    e = np.ones(ny - 1)
    d_dy = spdiags([-e, e], [0, 1], ny - 1, ny)
    
    A = csc_matrix((0, 0))

    for v in range(grid_size[4]):
        for u in range(grid_size[3]):
            for k in range(grid_size[2]):
                for j in range(grid_size[1]):
                    A = block_diag([A, d_dy]) 
    
    return A


def buildDerivZMatrix(grid_size):
    # d/dz for every entry in the cube except the last slice.
    m = grid_size[0] * grid_size[1] * (grid_size[2] - 1)
    n = grid_size[0] * grid_size[1] * grid_size[2]
    e = np.ones(m)
    d_dz = spdiags([-e, e], [0, grid_size[0] * grid_size[1]], m, n)
    
    A = csc_matrix((0, 0))

    for v in range(grid_size[4]):
        for u in range(grid_size[3]):
            A = block_diag([A, d_dz])

    return A


def buildSecondDerivZMatrix(grid_size):
    # d/dz for every entry in the cube except the first and last slice
    m = grid_size[0] * grid_size[1] * (grid_size[2] - 2)
    n = grid_size[0] * grid_size[1] * grid_size[2]
    e = np.ones(m)
    
    # Interior part of the matrix (central slices)
    interior = spdiags([e, -2 * e, e], [0, grid_size[0] * grid_size[1], 2 * grid_size[0] * grid_size[1]], m, n)
    
    # Boundary slices
    boundary_z1 = makeBoundaryZ1(grid_size, n)
    boundary_zend = makeBoundaryZEnd(grid_size, n)
    
    # The matrix for a full cube
    cube = vstack([boundary_z1, interior, boundary_zend])
    
    # Repeat the cube for the last 2 dimensions
    A = csc_matrix((0, 0))  # Initialize empty sparse matrix
    
    for v in range(grid_size[4]):  
        for u in range(grid_size[3]):  
            A = block_diag([A, cube])

    return A


def makeBoundaryZ1(grid_size, n):
    # Boundary conditions for the first slice
    mm = grid_size[0] * grid_size[1]
    nn = grid_size[0] * grid_size[1] * 2
    e = np.ones(mm)
    B = spdiags([-e, e], [0, grid_size[0] * grid_size[1]], mm, nn)
    
    # Concatenate zero columns to the right to match the full matrix later
    A = hstack([B, csc_matrix((mm, n - nn))])  # sparse matrix hstack

    return A


def makeBoundaryZEnd(grid_size, n):
    # Boundary conditions for the last slice
    mm = grid_size[0] * grid_size[1]
    nn = grid_size[0] * grid_size[1] * 2
    e = np.ones(mm)
    B = spdiags([e, -e], [0, grid_size[0] * grid_size[1]], mm, nn)
    
    # Concatenate zero columns to the left to match the full matrix later
    A = hstack([csc_matrix((mm, n - nn)), B])

    return A


def Fit(input_image, edge_image, output_image):
    
    DEFAULT_LAMBDA_SPATIAL = 1
    DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z = 4e-7
    
    # Check data types
    if not isinstance(input_image, np.ndarray) or input_image.dtype != np.float64:
        raise ValueError('input_image must be a double precision (float64) numpy array.')
    
    if not isinstance(output_image, np.ndarray) or output_image.dtype != np.float64:
        raise ValueError('output_image must be a double precision (float64) numpy array.')
    
    if edge_image is None or edge_image.ndim != 2 or edge_image.dtype != np.float64:
        raise ValueError('edge_image must be a double precision (float64) matrix (one channel).')

    # Ensure input and output dimensions match
    if input_image.shape[:2] != output_image.shape[:2]:
        raise ValueError('input_image and output_image must have the same width and height.')

    output_weight = np.ones_like(output_image)

    # Get default grid size
    grid_size = getDefaultAffineGridSize(input_image, output_image)
    lambda_spatial = DEFAULT_LAMBDA_SPATIAL

    # Default intensity_options
    intensity_options = {
        'type': 'second',
        'value': 0,
        'lambda': DEFAULT_SECOND_DERIVATIVE_LAMBDA_Z,
        'enforce_monotonic': False
    }

    # Calculate sizes and parameters
    input_height, input_width = input_image.shape[:2]
    grid_height, grid_width, grid_depth, affine_output_size, affine_input_size = grid_size

    bin_size_x = input_width / grid_width
    bin_size_y = input_height / grid_height
    bin_size_z = 1 / grid_depth

    num_deriv_y_rows = (grid_height - 1) * grid_width * grid_depth * affine_output_size * affine_input_size
    num_deriv_x_rows = grid_height * (grid_width - 1) * grid_depth * affine_output_size * affine_input_size

    # Initialize weight and slice matrices
    weight_matrices = [[None for _ in range(affine_input_size)] for _ in range(affine_output_size)]
    slice_matrices = [[None for _ in range(affine_input_size)] for _ in range(affine_output_size)]

    # Build slice matrices for each (i, j) entry of the affine model
    for j in range(affine_input_size):
        for i in range(affine_output_size):
            # Building weight and slice matrices, i = {i+1}, j = {j+1}
            weight_matrices[i][j], slice_matrices[i][j] = buildAffineSliceMatrix(input_image, edge_image, grid_size, i+1, j+1)

    # Check the expected number of columns based on one of the matrices
    num_columns = slice_matrices[0][0].shape[1] if slice_matrices[0][0].shape[1] > 0 else 0

    # Concatenate them together
    slice_matrix = csc_matrix((0, num_columns))
    weight_matrix = csc_matrix((0, 0))

    for j in range(affine_input_size):
        for i in range(affine_output_size):
            # Concatenating affine slice matrices, i = {i+1}, j = {j+1}
            slice_matrix = vstack([slice_matrix, slice_matrices[i][j]])
            weight_matrix = block_diag([weight_matrix, weight_matrices[i][j]])

    # Building apply affine model matrix
    apply_affine_model_matrix = buildApplyAffineModelMatrix(input_image, affine_output_size)

    # Building full slice matrix
    sqrt_w = np.sqrt(output_weight.ravel())
    W_data = spdiags(sqrt_w, 0, len(sqrt_w), len(sqrt_w))
    A_data = W_data @ apply_affine_model_matrix @ weight_matrix @ slice_matrix
    b_data = output_image.T.ravel() * sqrt_w

    # Building d/dy matrix
    A_deriv_y = (bin_size_x * bin_size_z / bin_size_y) * lambda_spatial * buildDerivYMatrix(grid_size)
    b_deriv_y = np.zeros(num_deriv_y_rows)

    # Building d/dx matrix
    A_deriv_x = (bin_size_y * bin_size_z / bin_size_x) * lambda_spatial * buildDerivXMatrix(grid_size)
    b_deriv_x = np.zeros(num_deriv_x_rows)

    # Building d/dz matrix (second derivative)
    A_intensity = (bin_size_x * bin_size_y) / (bin_size_z**2) * intensity_options['lambda'] * buildSecondDerivZMatrix(grid_size)
    b_intensity = intensity_options['lambda'] * intensity_options['value'] * np.ones(A_intensity.shape[0])

    # Assembling final sparse system
    if intensity_options['type'] != 'none':
        A = vstack([A_data, A_deriv_y, A_deriv_x, A_intensity]).tocsc()
        b = np.concatenate([b_data, b_deriv_y, b_deriv_x, b_intensity])
    else:
        A = vstack([A_data, A_deriv_y, A_deriv_x]).tocsc()
        b = np.concatenate([b_data, b_deriv_y, b_deriv_x])

    # Solve the system
    gamma = lsqr(A, b, iter_lim=500)[0]
    gamma = gamma.reshape(grid_size, order='F')

    return gamma


def bguSlice(gamma, input_fs):
    if isinstance(input_fs, np.ndarray):
        input_image = im2double(input_fs)
    else:
        input_image = np.transpose(np.squeeze(input_fs.cpu().numpy().astype(np.float64)), (1, 2, 0))
    
    edge_image = rgb2luminance(input_image)
    # Find downsampling coordinates, without rounding
    input_height, input_width = input_image.shape[:2]
    grid_height, grid_width, grid_depth, affine_output_size, affine_input_size = gamma.shape
    
    # meshgrid inputs and outputs are x, then y, with x right, y down
    x, y = np.meshgrid(np.arange(input_width), np.arange(input_height))
    
    # Downsample x and y to grid space
    bg_coord_x = ((x + 0.5) * (grid_width - 1) / input_width)
    bg_coord_y = ((y + 0.5) * (grid_height - 1) / input_height)
    bg_coord_z = edge_image * (grid_depth - 1)

    # Initialize affine_model
    from scipy.interpolate import RegularGridInterpolator
    affine_model = np.empty((affine_output_size, affine_input_size), dtype=object)
    for j in range(affine_input_size):
        for i in range(affine_output_size):
            interpolator = RegularGridInterpolator(
                (np.arange(grid_height), np.arange(grid_width), np.arange(grid_depth)),
                gamma[:, :, :, i, j], bounds_error=False, fill_value=None
            )
            coords = np.array([bg_coord_y.ravel(), bg_coord_x.ravel(), bg_coord_z.ravel()]).T
            affine_model[i, j] = interpolator(coords).reshape(input_height, input_width)

    # Combine results across affine_output_size
    affine_model2 = []
    for i in range(affine_output_size):
        affine_model2.append(np.stack([affine_model[i, j] for j in range(affine_input_size)], axis=-1))

    affine_model3 = np.stack(affine_model2, axis=-2)

    # Add the ones channel to the input image
    input1 = np.concatenate([input_image, np.ones((input_image.shape[0], input_image.shape[1], 1))], axis=-1)

    # Compute the final output as the sum of element-wise products
    output = np.zeros((input_height, input_width, affine_output_size))
    for i in range(affine_input_size):
        output += affine_model3[:, :, :, i] * input1[:, :, i][:, :, np.newaxis]
    
    return output


# Mapping input_fs to output_fs
def bguFit(input_fs, output_fs, bgu_ratio = 8):

    if isinstance(input_fs, np.ndarray):
        input_fs = im2double(input_fs)
    else:
        input_fs = np.transpose(np.squeeze(input_fs.cpu().numpy().astype(np.float64)), (1, 2, 0))
    
    if isinstance(output_fs, np.ndarray):
        output_fs = im2double(output_fs)
    else:
        output_fs = np.transpose(np.squeeze(output_fs.cpu().numpy().astype(np.float64)), (1, 2, 0))

    fh, fw = input_fs.shape[:2]

    input_ds = cv2.resize(input_fs, (fw//bgu_ratio, fh//bgu_ratio), interpolation=cv2.INTER_LANCZOS4) # INTER_LINEAR, INTER_CUBIC, INTER_AREA 
    output_ds = cv2.resize(output_fs, (fw//bgu_ratio, fh//bgu_ratio), interpolation=cv2.INTER_LANCZOS4)
    
    edge_ds = rgb2luminance(input_ds)
    
    gamma = Fit(input_ds, edge_ds, output_ds)
    return gamma


if __name__ == "__main__":
    input_fs = cv2.imread('high_in.png')
    output_fs = cv2.imread('high_out.png')
    gamma = bguFit(input_fs, output_fs)
    result_fs = bguSlice(gamma, input_fs)
    result = (np.clip(result_fs, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite('result.jpg',result)
