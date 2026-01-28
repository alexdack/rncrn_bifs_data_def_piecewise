import jax
import jax.numpy as jnp
from jax import jacobian, grad
import numpy as np
import numpy.ma as ma
from datetime import datetime
import scipy.io as sio
import optax

def initialize_single_RNCRN(number_of_exec_species, number_of_chemical_perceptrons, rnd_seed):
    jaxkey = jax.random.key(rnd_seed)
    W_key, b_key, alpha_key, beta_key, gamma_key, tau_key = jax.random.split(jaxkey, 6)
    alpha_mat = jax.random.normal(alpha_key, (number_of_exec_species, number_of_chemical_perceptrons))
    omega_mat = jax.random.normal(W_key, (number_of_chemical_perceptrons, number_of_exec_species))
    bias_vec = jax.random.normal(b_key, (number_of_chemical_perceptrons, 1))
    beta_vec = jnp.abs(jax.random.normal(beta_key, (number_of_exec_species, 1)))
    gamma_vec = jnp.abs(jax.random.normal(gamma_key, (number_of_chemical_perceptrons, 1)))
    tau_vec = jnp.abs(jax.random.normal(tau_key, (number_of_chemical_perceptrons, 1)))
    return (alpha_mat, omega_mat, bias_vec, beta_vec, gamma_vec, tau_vec)




# PURE FN implementation of the RNCRN loss with MSE
def quasi_static_loss_pure_fn(inputs, targets, alpha_mat, omega_mat, bias_vec, beta_vec, gamma_vec, tau_vec):
    sums = jnp.matmul(omega_mat, inputs) + bias_vec
    non_linearity = (sums+jnp.sqrt(jnp.power(sums,2)+4*jnp.abs(gamma_vec)*jnp.abs(tau_vec)))/(2*jnp.abs(tau_vec))
    preds = jnp.abs(beta_vec) + inputs*jnp.matmul(alpha_mat, non_linearity)
    return jnp.mean(jnp.square(targets - preds))

# PURE FN implementation of the static executive species RNCRN loss with MSE
def quasi_static_loss_static_execs(inputs, targets, static_execs, alpha_mat, omega_mat, bias_vec, beta_vec, gamma_vec, tau_vec, static_omega_mat):
    sums = jnp.matmul(omega_mat, inputs) + bias_vec + jnp.matmul(static_omega_mat, static_execs)
    non_linearity = (sums+jnp.sqrt(jnp.power(sums,2)+4*jnp.abs(gamma_vec)*jnp.abs(tau_vec)))/(2*jnp.abs(tau_vec))
    preds = jnp.abs(beta_vec) + inputs*jnp.matmul(alpha_mat, non_linearity)
    return jnp.mean(jnp.square(targets - preds))

def clean_params(params):
    return (params[0], params[1], params[2], jnp.abs(params[3]), jnp.abs(params[4]), jnp.abs(params[5]))

def train_basic_RNCRN(inputs, targets, params, number_of_epochs, start_learning_rate, batch_size, error_threshold):
    optimizer = optax.adamax(start_learning_rate)
    opt_state = optimizer.init(params)

    lowest_loss = 10000000        
    lowest_params = params

    for i in np.arange(1,number_of_epochs):
        grads = grad(quasi_static_loss_pure_fn, argnums=(2,3,4,5,6,7))(inputs, targets, params[0], params[1], params[2], params[3], params[4], params[5])
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
                
        if i % batch_size == 0:
            loss = quasi_static_loss_pure_fn(inputs, targets, params[0], params[1], params[2], params[3], params[4], params[5])

            print('Epoch:', i, 'loss = ', loss)
            
            if loss < lowest_loss:
                print('Lowest loss found!')
                lowest_loss = loss 
                lowest_params = params

            if loss < error_threshold:
                print('Solution FOUND!')
                break
   
    lowest_params = clean_params(lowest_params)
    params = clean_params(params)
    return (loss, params, lowest_params, lowest_loss)


def clean_params_static_execs(params):
    return (params[0], params[1], params[2], jnp.abs(params[3]), jnp.abs(params[4]), jnp.abs(params[5]), params[6])

def train_static_executive_species(inputs, targets, exec_inputs, params, number_of_epochs, start_learning_rate, batch_size, error_threshold):

    optimizer = optax.adamax(start_learning_rate)
    opt_state = optimizer.init(params)

    lowest_loss = 10000000        
    lowest_params = params

    for i in np.arange(1,number_of_epochs):
        grads = grad(quasi_static_loss_static_execs, argnums=(3,4,5,6,7,8,9))(inputs, targets, exec_inputs,  params[0], params[1], params[2], params[3], params[4], params[5], params[6])
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
                
        if i % batch_size == 0:
            loss = quasi_static_loss_static_execs(inputs, targets, exec_inputs, params[0], params[1], params[2], params[3], params[4], params[5], params[6])

            print('Epoch:', i, 'loss = ', loss)
            
            if loss < lowest_loss:
                print('Lowest loss found!')
                lowest_loss = loss 
                lowest_params = params

            if loss < error_threshold:
                print('Solution FOUND!')
                break
   
    lowest_params = clean_params_static_execs(lowest_params)
    params = clean_params_static_execs(params)
    return (loss, params, lowest_params, lowest_loss)

def save_mat_model_static_exec(params, number_of_chemical_perceptrons, number_of_static_exec_species, number_of_exec_species, loss, rnd_seed, flag='toogle', file='models'):
    dt = datetime.now()
    filename = file +'/model_static_exec_'+'n_' + str(number_of_chemical_perceptrons) + '_p_'+ str(number_of_static_exec_species) + '_'+ dt.strftime("%Y%m%d%H%M%S") + '_' + flag
    sio.savemat(filename+'.mat', {'alpha_mat':params[0], 
                                    'omega_mat':params[1], 
                                    'bias_vec':params[2],
                                    'beta': jnp.abs(params[3]),
                                    'gamma': jnp.abs(params[4]),
                                    'tau': jnp.abs(params[5]),                                     
                                    'static_omega_mat': params[6],                                                                      
                                    'loss': loss,
                                    'number_of_exec_species': number_of_exec_species,
                                    'number_chemical_perceptrons': number_of_chemical_perceptrons, 
                                    'number_of_static_exec_species': number_of_static_exec_species,
                                    'rnd_seed': rnd_seed,
                                    'loss_type': 'MSE'
                                    })
    
def unpack_mat_model_static_exec(filename):
    mat_contents = sio.loadmat(filename)
    params = (mat_contents['alpha_mat'], mat_contents['omega_mat'], mat_contents['bias_vec'], mat_contents['beta'], mat_contents['gamma'], mat_contents['tau'], mat_contents['static_omega_mat'])
    number_of_exec_species = mat_contents['number_of_exec_species']
    number_chemical_perceptrons = mat_contents['number_chemical_perceptrons']
    number_of_static_exec_species = mat_contents['number_of_static_exec_species']
    return params, number_of_exec_species[0][0], number_chemical_perceptrons[0][0], number_of_static_exec_species[0][0]

