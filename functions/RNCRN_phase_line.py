import numpy as np
import jax.numpy as jnp

# function that generates training data for a 1D executive system in the required format
def create_training_data_1D_from_ODE(dynamics_func, lower_limit, upper_limit, step_size ):
    x12_train = np.arange(lower_limit, upper_limit + step_size, step_size )
    inputs = np.array([x12_train])
    targets = dynamics_func(inputs)
    return (inputs, targets)

def compute_quasi_static_line_over_state_space_static_execs(inputs, static_execs, alpha_mat, omega_mat, bias_vec, beta_vec, gamma_vec, tau_vec, static_omega_mat):
    sums = jnp.matmul(omega_mat, inputs) + bias_vec + jnp.matmul(static_omega_mat, static_execs)
    non_linearity = (sums+jnp.sqrt(jnp.power(sums,2)+4*jnp.abs(gamma_vec)*jnp.abs(tau_vec)))/(2*jnp.abs(tau_vec))
    return jnp.abs(beta_vec) + inputs*jnp.matmul(alpha_mat, non_linearity)
