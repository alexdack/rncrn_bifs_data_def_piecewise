import numpy as np
import csv
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import scipy.io as sio
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.transforms as mtransforms
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.lines as mlines
from datetime import datetime


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 30
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
plt.rcParams['axes.prop_cycle'] = cycler(alpha=[1])
mpl.rcParams['figure.figsize'] = [12 , 5]
mpl.rcParams['text.usetex'] = False
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.edgecolor'] = 'k'
mpl.rcParams['legend.framealpha'] = 1
mpl.rcParams['legend.fancybox'] = False



# Function to open .csv files and extract them for plotting
def open_csv(file):
    out_data = []
    with open(file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in data:
            out_data.append([float(datapoint) for datapoint in row[0].split(',')])
    out_data=np.asarray(out_data) 
    return out_data

# Function to open compute the quasi-static approximation given a set of RNCRN params
def compute_quasi_dynamics(t, p, alpha_mat, omega_mat, bias_vec, beta, gamma, tau, number_of_exec_species, number_chemical_perceptrons): 
    xs = p.reshape(number_of_exec_species,1)
    sums = np.add(np.matmul(omega_mat, xs),bias_vec)
    ys = (sums + np.sqrt(np.power(sums,2) + 4*gamma*tau))/(2*tau)
    dx = beta +  xs*np.matmul(alpha_mat,ys)
    return list(dx.reshape(number_of_exec_species,))

# Function for extracting weights from .mat RNCRN model as a ordered tuple
def unpack_mat_model_rncrn(filename):
    mat_contents = sio.loadmat(filename)
    params = (mat_contents['alpha_mat'], mat_contents['omega_mat'], mat_contents['bias_vec'], mat_contents['beta'], mat_contents['gamma'], mat_contents['tau'])
    number_of_exec_species = mat_contents['number_of_exec_species']
    number_chemical_perceptrons = mat_contents['number_chemical_perceptrons']
    return params, number_of_exec_species[0][0], number_chemical_perceptrons[0][0]

# Function to save CRN.txt files
def save_crn(filename, txt):
    f = open(filename, "w")
    f.write(txt)
    f.close()

# Function to parse CRN.txt files to use in python 
def read_crn_txt(filename):
    f = open(filename, "r")
    species_and_nothing = {''}
    reaction_rates = []
    reaction_reacts = []
    reaction_prods = []
    initial_concs = {}
    for x in f:
        if x.startswith("#"):
            conc_string = x.replace(' ', '').strip(' \n\t#')
            split_concs = conc_string.split(",")
            for conc in split_concs:
                species_name, conc_val = conc.split("=")
                if species_name in initial_concs:
                    conc_val_stored = initial_concs[species_name]
                    max_conc = max(conc_val_stored, float(conc_val))
                    initial_concs[species_name]= float(max_conc)
                else:
                    initial_concs.update({species_name: float(conc_val)})
        else:
            x = x.replace(' ', '').strip(' \n\t')
            split_comma = x.split(",")
            rate = float(split_comma[1].strip(' \n\t'))
            split_arrow = split_comma[0].split("->")
            reactants = split_arrow[0].split("+")
            products = split_arrow[1].split("+")

            reaction_rates = reaction_rates + [rate]

            for reactant in reactants:
                species_and_nothing.add(reactant)

            for product in products:
                species_and_nothing.add(product)

            reaction_reacts = reaction_reacts + [reactants]
            reaction_prods = reaction_prods + [products]
    
    number_species = len(species_and_nothing) - 1 
    species = list(species_and_nothing)[1:]
    number_reactions = len(reaction_rates)
    
    react_stoch = np.zeros(shape=(number_reactions, number_species))
    prod_stoch = np.zeros(shape=(number_reactions, number_species))

    for r in np.arange(0, number_reactions):
        reacts_for_react = reaction_reacts[r]
        prods_for_react = reaction_prods[r]
        for s in np.arange(0, number_species):
            react_stoch[r,s] = reacts_for_react.count(species[s])
            prod_stoch[r,s] = prods_for_react.count(species[s])

    stoch_mat = prod_stoch - react_stoch
    
    initial_concs_vec = np.zeros(shape=(number_species,))
    for species_idx in np.arange(0,len(species)):
        if species[species_idx] in initial_concs:
            initial_concs_vec[species_idx] = initial_concs[species[species_idx]]

    return (species, reaction_rates, react_stoch, prod_stoch, stoch_mat, number_species, number_reactions, initial_concs_vec )

def new_initial_conditions(old_inits, species, dict):
    new_inits = []
    id_dict = {}
    for s in np.arange(0, len(species), 1):
        if species[s] in dict.keys():
            new_inits += [dict[species[s]]]
        else:
            new_inits += [old_inits[s]]
        id_dict[species[s]] = s
    return new_inits, id_dict

# Converts a RNCRN into a CRN (string)
def convert_RNCRN_params_to_CRN(params, time_scale, inits):
    alpha_mat, omega_mat, bias_vec, beta, gamma, tau = params

    CRN = '#'

    number_of_executive_species, number_of_chemical_perceptrons = alpha_mat.shape

    for i in np.arange(0, number_of_executive_species):
        CRN = CRN + 'X_'+ str(i+1) + '='+ str(inits[i]) +','

    for j in np.arange(0, number_of_chemical_perceptrons):
        CRN = CRN + 'Y_'+ str(j+1) + '='+ str(inits[number_of_executive_species + j]) +','

    CRN = CRN[:-1]
    CRN += '\n'

    dim_alpha = alpha_mat.shape
    for i in np.arange(0, dim_alpha[0]):
        for j in np.arange(0, dim_alpha[1]):
            xstr = 'X_' + str(i+1) 
            ystr = 'Y_' + str(j+1)
            rate = str(abs(alpha_mat[i,j]))
            if alpha_mat[i,j] > 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + ystr +  ' + ' + xstr + ' + ' + xstr + ',' + rate +'\n'
            elif alpha_mat[i,j] < 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + ystr + ',' + rate +'\n'

    dim_omega= omega_mat.shape
    for i in np.arange(0, dim_omega[1]):
        for j in np.arange(0, dim_omega[0]):
            xstr = 'X_' + str(i+1) 
            ystr = 'Y_' + str(j+1)
            rate = str(abs(omega_mat[j,i])/time_scale)
            if omega_mat[j,i] > 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr +  ' + ' + ystr + ' + ' + ystr + ',' + rate +'\n'
            elif omega_mat[j,i] < 0:
                CRN = CRN + xstr + ' + ' + ystr + '->' + xstr + ',' + rate +'\n'

    dim_bias= bias_vec.shape
    for i in np.arange(0, dim_bias[0]):
        ystr = 'Y_' + str(i+1)
        rate = str(abs(bias_vec[i][0])/time_scale)
        if bias_vec[i] > 0:
            CRN = CRN + ystr + '->' + ystr + ' + ' + ystr + ',' + rate +'\n'
        elif bias_vec[i] < 0:
            CRN = CRN + ystr + '->' + ',' + rate +'\n'

    dim_beta= beta.shape
    for i in np.arange(0, dim_beta[0]):
        xstr = 'X_' + str(i+1)
        rate = str(abs(beta[i][0]))
        CRN = CRN + '->' + xstr + ',' + rate +'\n'

    dim_gamma= gamma.shape
    for i in np.arange(0, dim_gamma[0]):
        ystr = 'Y_' + str(i+1)
        rate = str(abs(gamma[i][0])/time_scale)
        CRN = CRN + '->' + ystr + ',' + rate +'\n'

    dim_tau= tau.shape
    for i in np.arange(0, dim_tau[0]):
        ystr = 'Y_' + str(i+1)
        rate = str(abs(tau[i][0])/time_scale)
        CRN = CRN + ystr + ' + ' +ystr +'->' + ystr + ',' + rate +'\n'
    
    return CRN

# Converts a RNCRN with static execs into a CRN (string)
def convert_static_exec_RNCRN_params_to_CRN(params, time_scale, inits, control_inits, number_of_exec_species, number_of_chemical_perceptrons, number_of_control_params):
    alpha_mat, omega_mat_full, bias_vec, beta, gamma, tau = params
    omega_mat = omega_mat_full[:,:number_of_exec_species]
    static_omega_mat = omega_mat_full[:,number_of_exec_species:]

    new_params = (alpha_mat, omega_mat, bias_vec, beta, gamma, tau)

    number_of_static_exec_species = number_of_control_params
    CRN = '#'
    for i in np.arange(0, number_of_control_params):
        lp_init = control_inits[i]
        CRN = CRN + 'L_'+ str(i+1) + '='+ str(lp_init) +','
    CRN = CRN[:-1]
    CRN += '\n'

    dim_omega= static_omega_mat.shape
    for i in np.arange(0, dim_omega[1]):
        for j in np.arange(0, dim_omega[0]):
            lpstr = 'L_' + str(i+1) 
            ystr = 'Y_' + str(j+1)
            rate = str(abs(static_omega_mat[j,i])/time_scale)
            if static_omega_mat[j,i] > 0:
                CRN = CRN + lpstr + ' + ' + ystr + '->' + lpstr +  ' + ' + ystr + ' + ' + ystr + ',' + rate +'\n'
            elif static_omega_mat[j,i] < 0:
                CRN = CRN + lpstr + ' + ' + ystr + '->' + lpstr + ',' + rate +'\n'
    return CRN + convert_RNCRN_params_to_CRN(new_params, time_scale, inits)

# Converts a RNCRN with static execs into a CRN (string) - OLD
def convert_static_exec_RNCRN_params_to_CRN_not_shifted(params, time_scale, inits, control_inits, number_of_exec_species, number_of_chemical_perceptrons, number_of_control_params, ref_conc=0):
    alpha_mat, omega_mat_full, bias_vec, beta, gamma, tau = params
    omega_mat = omega_mat_full[:,:number_of_exec_species]
    static_omega_mat = omega_mat_full[:,number_of_exec_species:]

    new_params = (alpha_mat, omega_mat, bias_vec, beta, gamma, tau)

    number_of_static_exec_species = 2*number_of_control_params
    CRN = '#'
    for i in np.arange(0, number_of_control_params):
        if control_inits[i] > 0:
            lp_init = control_inits[i] + ref_conc
            lm_init = ref_conc
        else:
            lp_init =  ref_conc
            lm_init = control_inits[i] + ref_conc

        CRN = CRN + 'LP_'+ str(i+1) + '='+ str(lp_init) +','
        CRN = CRN + 'LM_'+ str(i+1) + '='+ str(lm_init) +','

    CRN = CRN[:-1]
    CRN += '\n'


    dim_omega= static_omega_mat.shape
    for i in np.arange(0, dim_omega[1]):
        for j in np.arange(0, dim_omega[0]):
            lpstr = 'LP_' + str(i+1) 
            lmstr = 'LM_' + str(i+1) 
            ystr = 'Y_' + str(j+1)
            rate = str(abs(static_omega_mat[j,i])/time_scale)
            if static_omega_mat[j,i] > 0:
                CRN = CRN + lpstr + ' + ' + ystr + '->' + lpstr +  ' + ' + ystr + ' + ' + ystr + ',' + rate +'\n'
                CRN = CRN + lmstr + ' + ' + ystr + '->' + lmstr + ',' + rate +'\n'
            elif static_omega_mat[j,i] < 0:
                CRN = CRN + lpstr + ' + ' + ystr + '->' + lpstr + ',' + rate +'\n'
                CRN = CRN + lmstr + ' + ' + ystr + '->' + lmstr +  ' + ' + ystr + ' + ' + ystr + ',' + rate +'\n'

    return CRN + convert_RNCRN_params_to_CRN(new_params, time_scale, inits)
          
# Function that converts a stoichiometry matrix into a reaction-rate equation 
def stoch_mat_to_mass_action(t,x, reaction_rates, react_stoch, stoch_mat):
    conc_to_power_of_react = np.power(x, react_stoch);
    fluxes = np.prod(conc_to_power_of_react, axis=1)
    fluxes_with_rates = np.asarray(reaction_rates)*fluxes
    mass_action = np.matmul(np.transpose(stoch_mat), fluxes_with_rates)
    return mass_action

# Function for checking if peaks in frequency domain
def peak_check(y, N=600, tFinal = 60 ):
    T = tFinal / N
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    freq_domain = 2.0/N * np.abs(yf[0:N//2])
    peaks, _ = find_peaks(freq_domain, height=0.5)
    return len(peaks) > 0 

def save_mat_model_basic_rncrn(params, number_of_chemical_perceptrons, number_of_exec_species, loss, rnd_seed, flag='', file='models'):
    dt = datetime.now()
    filename = file +'/model_basic_rncrn_'+'n_' + str(number_of_chemical_perceptrons) + '_'+ dt.strftime("%Y%m%d%H%M%S") + '_' + flag
    sio.savemat(filename+'.mat', {'alpha_mat':params[0], 
                                    'omega_mat':params[1], 
                                    'bias_vec':params[2],
                                    'beta': np.abs(params[3]),
                                    'gamma': np.abs(params[4]),
                                    'tau': np.abs(params[5]),                                     
                                    'loss': loss,
                                    'number_of_exec_species': number_of_exec_species,
                                    'number_chemical_perceptrons': number_of_chemical_perceptrons, 
                                    'rnd_seed': rnd_seed,
                                    'loss_type': 'MSE'
                                    })

# function to load tf model that has been saved as .mat to be used by the new JAX layout
def load_tf_models_params(file, beta_val=1, gamma_val=1, tau_val=1):
    mat_contents = sio.loadmat(file)
    alpha_mat = mat_contents['output_layer_weights']
    omega_mat = mat_contents['first_layer_weights']
    bias_vec = mat_contents['first_layer_biases']

    number_of_chemical_perceptrons = bias_vec.shape[1]
    number_of_exec_species = alpha_mat.shape[1]
    loss = False
    rnd_seed = False

    try:
        beta_val_fil =  mat_contents['beta'][0][0]
        beta = beta_val_fil*np.ones(shape=(number_of_exec_species,1))
    except:
        beta = beta_val*np.ones(shape=(number_of_exec_species,1))
    
    try:
        gamma_val_fil = mat_contents['gamma'][0][0]
        gamma = gamma_val_fil*np.ones(shape=(number_of_chemical_perceptrons,1))
    except:
        gamma = gamma_val*np.ones(shape=(number_of_chemical_perceptrons,1))
    
    try:
        tau_fil = mat_contents['alpha'][0][0]
        tau = tau_fil*np.ones(shape=(number_of_chemical_perceptrons,1))
    except:
        tau = tau_val*np.ones(shape=(number_of_chemical_perceptrons,1))

    alpha_mat = alpha_mat.transpose()
    omega_mat = omega_mat.transpose()
    bias_vec = bias_vec.transpose()

    params =  (alpha_mat, omega_mat, bias_vec, beta,gamma, tau) 

    return params, number_of_exec_species, number_of_chemical_perceptrons