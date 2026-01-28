import numpy as np
from scipy.io import loadmat
from functions.RNCRN_tools import convert_RNCRN_params_to_CRN
from functions.RNCRN_train import unpack_mat_model_static_exec

def load_sac_classify(filename, taus = 1):
    classify_layer_contents = loadmat(filename)
    omega_env_2_layer = np.transpose(classify_layer_contents['first_layer_weights'])
    theta_env_2_layer = np.transpose(classify_layer_contents['first_layer_biases'])
    omega_layer_2_bif = np.transpose(classify_layer_contents['output_layer_weights'])
    theta_layer_2_bif = np.transpose(classify_layer_contents['output_layer_biases'])
    gammas = classify_layer_contents['gamma']

    gammas_env = gammas[0,0]*np.ones(theta_env_2_layer.shape)
    taus_env = taus*np.ones(theta_env_2_layer.shape)

    gammas_bif = gammas[0,0]*np.ones(omega_layer_2_bif.shape)
    taus_bif = taus*np.ones(omega_layer_2_bif.shape)

    return (omega_env_2_layer, theta_env_2_layer, gammas_env, taus_env, omega_layer_2_bif, theta_layer_2_bif, gammas_bif, taus_bif )

def convert_classify_to_CRN(params, inits, time_scale):
    omega_env_2_layer, theta_env_2_layer, gammas_env, taus_env, omega_layer_2_bif, theta_layer_2_bif, gammas_bif, taus_bif = params
    
    CRN = '#'
    number_of_env_inputs = omega_env_2_layer.shape[1]
    number_of_chemical_perceptrons = omega_env_2_layer.shape[0]
    number_of_bif_params = len(theta_layer_2_bif)

    for i in np.arange(0, number_of_env_inputs):
        CRN = CRN + 'L_'+ str(i+1) + '='+ str(inits[i]) +','

    for j in np.arange(0, number_of_chemical_perceptrons):
        CRN = CRN + 'Z_'+ str(j+1) + '='+ str(inits[number_of_env_inputs + j]) +','
        
    for k in np.arange(0, number_of_bif_params):
        CRN = CRN + 'R_'+ str(k+1) + '='+ str(inits[number_of_env_inputs + number_of_chemical_perceptrons + k]) +','

    CRN = CRN[:-1]
    CRN += '\n'

    dim_omega_env = omega_env_2_layer.shape
    for j in np.arange(0, dim_omega_env[1]):
        for i in np.arange(0, dim_omega_env[0]):
            ystr = 'Z_' + str(i+1) 
            xstr = 'L_' + str(j+1)
            rate = str(abs(omega_env_2_layer[i,j])/time_scale)
            if omega_env_2_layer[i,j] > 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr +  ' + ' + ystr + ' + ' + ystr + ',' + rate +'\n'
            elif omega_env_2_layer[i,j] < 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr + ',' + rate +'\n'
    
    dim_bias_env = theta_env_2_layer.shape

    for i in np.arange(0, dim_bias_env[0]):
        ystr = 'Z_' + str(i+1)
        rate = str(abs(theta_env_2_layer[i,0])/time_scale)
        if theta_env_2_layer[i] > 0:
            CRN = CRN + ystr + '->' + ystr + ' + ' + ystr + ',' + rate +'\n'
        elif theta_env_2_layer[i] < 0:
            CRN = CRN + ystr + '->' + ',' + rate +'\n'

    dim_gamma_env= gammas_env.shape
    for i in np.arange(0, dim_gamma_env[0]):
        ystr = 'Z_' + str(i+1)
        rate = str(abs(gammas_env[i][0])/time_scale)
        CRN = CRN + '->' + ystr + ',' + rate +'\n'

    dim_tau_env= taus_env.shape
    for i in np.arange(0, dim_tau_env[0]):
        ystr = 'Z_' + str(i+1)
        rate = str(abs(taus_env[i][0])/time_scale)
        CRN = CRN + ystr + ' + ' +ystr +'->' + ystr + ',' + rate +'\n'
        
    # next layer
    dim_omega_bif = omega_layer_2_bif.shape
    for j in np.arange(0, dim_omega_bif[1]):
        for i in np.arange(0, dim_omega_bif[0]):
            ystr = 'R_' + str(i+1) 
            xstr = 'Z_' + str(j+1)
            rate = str(abs(omega_layer_2_bif[i,j])/time_scale)
            if omega_layer_2_bif[i,j] > 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr +  ' + ' + ystr + ' + ' + ystr + ',' + rate +'\n'
            elif omega_layer_2_bif[i,j] < 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr + ',' + rate +'\n'
    
    dim_bias_bif = theta_layer_2_bif.shape
    for i in np.arange(0, dim_bias_bif[0]):
        ystr = 'R_' + str(i+1)
        rate = str(abs(theta_layer_2_bif[i,0])/time_scale)
        if theta_layer_2_bif[i] > 0:
            CRN = CRN + ystr + '->' + ystr + ' + ' + ystr + ',' + rate +'\n'
        elif theta_layer_2_bif[i] < 0:
            CRN = CRN + ystr + '->' + ',' + rate +'\n'

    dim_gamma_bif= gammas_bif.shape
    for i in np.arange(0, dim_gamma_bif[0]):
        ystr = 'R_' + str(i+1)
        rate = str(abs(gammas_bif[i][0])/time_scale)
        CRN = CRN + '->' + ystr + ',' + rate +'\n'

    dim_tau_bif= taus_bif.shape
    for i in np.arange(0, dim_tau_bif[0]):
        ystr = 'R_' + str(i+1)
        rate = str(abs(taus_bif[i][0])/time_scale)
        CRN = CRN + ystr + ' + ' +ystr +'->' + ystr + ',' + rate +'\n'
    return CRN

def convert_static_exec_RNCRN_params_to_CRN_modular(params, time_scale, inits, static_execs_inits, include_bif=True):
    alpha_mat, omega_mat, bias_vec, beta, gamma, tau, static_omega_mat = params
    new_params = ( alpha_mat, omega_mat, bias_vec, beta, gamma, tau)

    number_of_executive_species, _ = alpha_mat.shape
    _, number_of_static_executive_species = static_omega_mat.shape

    if include_bif:
        CRN = '#'

        for i in np.arange(0, number_of_static_executive_species):
            CRN = CRN + 'R_'+ str(i+1) + '='+ str(static_execs_inits[i]) +','
        CRN = CRN[:-1]
        CRN += '\n'
    else:
        CRN = ''

    dim_static_omega= static_omega_mat.shape
    for i in np.arange(0, dim_static_omega[1]):
        for j in np.arange(0, dim_static_omega[0]):
            xstr = 'R_' + str(i+1) 
            ystr = 'Y_' + str(j+1)
            rate = str(abs(static_omega_mat[j,i])/time_scale)
            if static_omega_mat[j,i] > 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr +  ' + ' + ystr + ' + ' + ystr + ',' + rate +'\n'
            elif static_omega_mat[j,i] < 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr + ',' + rate +'\n'
    
    return CRN + convert_RNCRN_params_to_CRN(new_params, time_scale, inits)

def create_sense_and_respond_CRN(filename_classify, filename_toggle, time_scale_classify, time_scale_toggle, env_inits, neuron_classify_inits, bif_inits,exec_species, neuron_toggle_inits):
    classify_inits = env_inits + neuron_classify_inits + bif_inits
    toggle_inits = exec_species + neuron_toggle_inits

    params_classify = load_sac_classify(filename_classify)
    params_toggle, _, _, _ = unpack_mat_model_static_exec(filename_toggle)

    CRN_classify = convert_classify_to_CRN(params_classify, classify_inits, time_scale_classify)
    CRN_toggle = convert_static_exec_RNCRN_params_to_CRN_modular(params_toggle, time_scale_toggle, toggle_inits, bif_inits, include_bif=False)
    return CRN_classify + CRN_toggle