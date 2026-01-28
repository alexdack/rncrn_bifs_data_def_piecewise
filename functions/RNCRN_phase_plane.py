import numpy as np

def meshgrid_to_array2D(xv1, xv2):
    return np.array([xv1.flatten(), xv2.flatten()])

def array2D_to_meshgrid(inputs, data_shape):
    return inputs.reshape(2, data_shape[0], data_shape[1])

def create_state_space_2D_array(lower_limit, upper_limit, step_size ):
    x12_train = np.arange(lower_limit, upper_limit + step_size, step_size )
    xv1, xv2 = np.meshgrid(x12_train, x12_train, indexing='ij')
    data_shape = xv1.shape
    return (meshgrid_to_array2D(xv1, xv2), data_shape)

def merge_multi_state_data(list_of_inputs, list_of_targets):
    if len(list_of_inputs) == len(list_of_targets) and len(list_of_inputs):
        inputs = np.concatenate(list_of_inputs, axis=1)
        targets = np.concatenate(list_of_targets, axis=1)
        return (inputs, targets)
    else:
        raise Exception('list_of_inputs and list_of_targets are not the same length.')

def compute_quasi_static_vector_field_over_state_space(inputs, alpha_mat, omega_mat, bias_vec, beta_vec, gamma_vec, tau_vec):
    sums = np.matmul(omega_mat, inputs) + bias_vec
    non_linearity = (sums+np.sqrt(np.power(sums,2)+4*np.abs(gamma_vec)*np.abs(tau_vec)))/(2*np.abs(tau_vec))
    return np.abs(beta_vec) + inputs*np.matmul(alpha_mat, non_linearity)

def compute_quasi_static_vector_field_over_state_space_static_exec(inputs, static_execs, alpha_mat, omega_mat, bias_vec, beta_vec, gamma_vec, tau_vec, static_omega_mat):
    sums = np.matmul(omega_mat, inputs) + bias_vec + np.matmul(static_omega_mat, static_execs)
    non_linearity = (sums+np.sqrt(np.power(sums,2)+4*np.abs(gamma_vec)*np.abs(tau_vec)))/(2*np.abs(tau_vec))
    return np.abs(beta_vec) + inputs*np.matmul(alpha_mat, non_linearity)
