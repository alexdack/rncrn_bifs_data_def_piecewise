import numpy as np
from scipy.linalg import qr

# basic function for creating the vector field between two points
def draw_padded_vector_field_two_points(first_point, second_point, distance, magnitude, num_padd, M=1, is_repeller=False, is_linear=False):
    # determine the normal direction from P1 to P2
    v = second_point-first_point
    v_norm = v / np.linalg.norm(v)
    norm_vec = np.matmul(np.array([[0, -1], [1,0]]), np.transpose(v) )
    unit_norm_vec = norm_vec / np.linalg.norm(norm_vec)

    stability_factor = 1 if not is_repeller else -1
    
    # records the vector field (x, \bar{x}) = (inputs, targets)
    inputs = np.array([first_point]);
    targets = np.array([magnitude*v_norm ]);

    # adds 2x'num_padd' many padding vectors around first_point which decay exponentially with distance 
    for i in range(1, num_padd+1):
        Q_plus = first_point + i*distance*unit_norm_vec
        Q_neg = first_point - i*distance*unit_norm_vec
        inputs = np.concatenate((inputs, np.array([Q_plus, Q_neg])))
        if is_linear:
            k2 = stability_factor*magnitude*M*i*distance
        else:
            k2 = stability_factor*magnitude*np.exp(M*i*distance)
        targets = np.concatenate((targets, np.array([magnitude*v_norm - k2*unit_norm_vec , magnitude*v_norm + k2*unit_norm_vec])))

    return (inputs, targets)

# draw attractor along points
def draw_attractor_points(points_arr, distance, magnitude, number_of_padding_each_side, final_point = True, is_repeller=False, old_fn=True, is_linear=False, M=1):
    N = points_arr.shape[0]
    number_of_dims = points_arr.shape[1]

    if number_of_dims < 2:
        raise ValueError("This only works for more than two dimensions.")

    data_point_per =  1 + 2*(number_of_padding_each_side)*(number_of_dims-1)
    res_input = np.zeros((data_point_per*N,number_of_dims))
    res_targets = np.zeros((data_point_per*N,number_of_dims))

    fn_draw_padded_vector_field_two_points = draw_padded_vector_field_two_points if old_fn else draw_padded_vector_field_two_points_high_dim

    for i in np.arange(0, N-1, 1):
        inputs, targets = fn_draw_padded_vector_field_two_points(points_arr[i], points_arr[i+1], distance, magnitude, number_of_padding_each_side, is_repeller=is_repeller, is_linear=is_linear, M=M)
        res_input[i*data_point_per:(i+1)*data_point_per, :] = inputs
        res_targets[i*data_point_per:(i+1)*data_point_per, :] = targets
    
    if final_point:
        inputs, targets = fn_draw_padded_vector_field_two_points(points_arr[N-1], points_arr[0], distance, magnitude, number_of_padding_each_side, is_repeller=is_repeller, is_linear=is_linear, M=M)
        res_input[(N-1)*data_point_per:N*data_point_per, :] = inputs
        res_targets[(N-1)*data_point_per:N*data_point_per, :] = targets 
    else:
        res_input = res_input[:(N-1)*data_point_per,:]
        res_targets = res_targets[:(N-1)*data_point_per,:]

    inputs_a = np.transpose(res_input)
    targets_a = np.transpose(res_targets)

    return inputs_a, targets_a

# draw attractor along parametric curve
def draw_attractor_parametric(fn_parametric, low_parametric, upper_parametric, step_parametric, distance, magnitude, number_of_padding_each_side, final_point = True, old_fn=True, is_linear=False, M=1):
    t_arr = np.arange(low_parametric, upper_parametric, step_parametric)
    points_arr = fn_parametric(t_arr)
    return draw_attractor_points(points_arr, distance, magnitude, number_of_padding_each_side, final_point = final_point, old_fn=old_fn, is_linear=is_linear, M=M)

# draw repeller along parametric curve
def draw_repeller_parametric(fn_parametric, low_parametric, upper_parametric, step_parametric, distance, magnitude, number_of_padding_each_side, final_point = True, old_fn=True, is_linear=False, M=1):
    t_arr = np.arange(low_parametric, upper_parametric, step_parametric)
    points_arr = fn_parametric(t_arr)
    return draw_attractor_points(points_arr, distance, magnitude, number_of_padding_each_side, final_point = final_point, is_repeller=True, old_fn=old_fn, is_linear=is_linear, M=M)

def compute_orthonormal_directions(forward_norm):
    dims = len(forward_norm)
    forward_norm_mat = forward_norm.reshape(dims, 1)
    Q, R = qr(forward_norm_mat)
    return Q, dims

def draw_padded_vector_field_two_points_high_dim(first_point, second_point, distance, magnitude, num_padd, M=1, is_repeller=False, is_linear=False):
    # determine the normal direction from P1 to P2
    v = second_point-first_point
    v_norm = v / np.linalg.norm(v)
    orth_mat, dims = compute_orthonormal_directions(v_norm)

    is_close = np.all(np.abs(v_norm - orth_mat[:,0]) < 1e-5)
    is_ops = np.all(np.abs(v_norm + orth_mat[:,0]) < 1e-5)
    
    if not is_close and not is_ops:
        print(v_norm == orth_mat[:,0])
        print(v_norm)
        print(orth_mat)
        raise ValueError("QR decomposition did not find basis with intended forward vector.")
    
    if is_ops:
        orth_mat = -1*orth_mat
    
    stability_factor = 1 if not is_repeller else -1
    
    # records the vector field (x, \bar{x}) = (inputs, targets)
    inputs = np.array([first_point]);
    targets = np.array([magnitude*v_norm ]);

    for dim in np.arange(1, dims, 1):
        unit_norm_vec = orth_mat[:,dim]
        # adds 2x'num_padd' many padding vectors around first_point which decay exponentially with distance 
        for i in range(1, num_padd+1):
            Q_plus = first_point + i*distance*unit_norm_vec
            Q_neg = first_point - i*distance*unit_norm_vec
            inputs = np.concatenate((inputs, np.array([Q_plus, Q_neg])))
            if is_linear:
                k2 = stability_factor*magnitude*M*i*distance
            else:
                k2 = stability_factor*magnitude*np.exp(M*i*distance)
            targets = np.concatenate((targets, np.array([magnitude*v_norm - k2*unit_norm_vec , magnitude*v_norm + k2*unit_norm_vec])))

    return (inputs, targets)