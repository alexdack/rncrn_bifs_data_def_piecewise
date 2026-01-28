import numpy as np
import jax
import jax.numpy as jnp

# function to create training data where one input is held at a constant value
def initialize_static_execs_data(number_of_static_exec_species, inputs, static_exec_state):
    if len(static_exec_state) == number_of_static_exec_species:
        np_arr = np.ones((number_of_static_exec_species, inputs.shape[1]))
        for i in np.arange(0,number_of_static_exec_species, 1):
            np_arr[i,:] = static_exec_state[i]*np_arr[i,:]
        return jnp.array(np_arr)
    else:
        raise Exception("number_of_static_exec_species and static_exec_state do not match in length.")
    
# inits the RNCRN (static execs) with a random starting paramater sets
def initialize_single_RNCRN_static_exec(number_of_exec_species, number_of_chemical_perceptrons, number_of_static_exec_species, rnd_seed):
    jaxkey = jax.random.key(rnd_seed)
    W_key, b_key, alpha_key, beta_key, gamma_key, tau_key, static_key = jax.random.split(jaxkey, 7)
    alpha_mat = jax.random.normal(alpha_key, (number_of_exec_species, number_of_chemical_perceptrons))
    omega_mat = jax.random.normal(W_key, (number_of_chemical_perceptrons, number_of_exec_species))
    bias_vec = jax.random.normal(b_key, (number_of_chemical_perceptrons, 1))
    beta_vec = jnp.abs(jax.random.normal(beta_key, (number_of_exec_species, 1)))
    gamma_vec = jnp.abs(jax.random.normal(gamma_key, (number_of_chemical_perceptrons, 1)))
    tau_vec = jnp.abs(jax.random.normal(tau_key, (number_of_chemical_perceptrons, 1)))
    static_omega_mat = jax.random.normal(static_key, (number_of_chemical_perceptrons, number_of_static_exec_species))
    return (alpha_mat, omega_mat, bias_vec, beta_vec, gamma_vec, tau_vec, static_omega_mat)

# merges executive data 
def merge_static_execs_multi_state_data(list_of_inputs, list_of_targets, list_of_exec_inputs):
    if len(list_of_inputs) == len(list_of_targets) and len(list_of_inputs) == len(list_of_exec_inputs):
        inputs = jnp.concatenate(list_of_inputs, axis=1)
        targets = jnp.concatenate(list_of_targets, axis=1)
        exec_inputs = jnp.concatenate(list_of_exec_inputs, axis=1)
        return (inputs, targets, exec_inputs)
    else:
        raise Exception('list_of_inputs, list_of_targets, and  list_of_exec_inputs are not the same length.')