import numpy as np

# ODE of the homoclinic bifurcation 
def homoclinic_bifurcation(t, y, a, r_critical, mu): 
    x1, x2 = y
    dx1 = (mu-r_critical-(4/5))*(x1-a)+(x2-a)-(6/5)*(x1-a)*(x2-a)+(3/2)*pow((x2-a),2)
    dx2 = (x1-a) -(4/5)*(x2-a) - (4/5)*pow((x2-a),2)
    return [dx1, dx2]

# ODE of supercritical Hopf bifurcation 
def supercritical_hopf_normal_form(t, y, a, r_critical, mu): 
    x1, x2 = y
    D = (mu-r_critical-pow(x1-a,2) - pow(x2-a,2))
    return [D*(x1-a) - (x2-a), D*(x2-a) + (x1-a)]

# parametric (t) representation of the lemniscate attractor
def parametric_lemniscate(t):
    return np.transpose(np.array([2+ (np.cos(t)/(np.sin(t)**2 +1)), 2+((np.cos(t)*np.sin(t))/(np.sin(t)**2 +1))]))

# parametric (t) representation of the heart attractor
def parametric_heart(t):
    a1 = 2
    a2 = 2
    b1 = 0.07
    b2 = 0.07
    return np.transpose(np.array([a1+ b1*(16*np.power(np.sin(t),3)), a2+b2*(13*np.cos(t)- 5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))]))

# parametric (t) representation of the helix attractor
def parametric_helix(t):
    return np.transpose(np.array([np.cos(t) + 2, np.sin(t) + 2, t]))

# parametric (t) representation of the torus attractor
def parametric_torus(t):
    return np.transpose(np.array([3*np.cos(t) +np.cos(10*t)*np.cos(t)+ 6, 3*np.sin(t) +np.cos(10*t)*np.sin(t)+ 6, np.sin(10*t)+2]))


def parametric_circle(t):
    return np.transpose(np.array([np.cos(t) + 2, np.sin(t) + 2]))

# parametric (t) representation of the circle attractor
def parametric_concentric_circle(t, radius, centre):
    return np.transpose(np.array([radius*np.cos(t) + centre, radius*np.sin(t) + centre]))

# parametric (t) representation of the circle attractor
def parametric_disjoint_cycle(t, radius, centre_1, centre_2):
    return np.transpose(np.array([radius*np.cos(t) + centre_1, radius*np.sin(t) + centre_2]))

# uni-stable linear function 
def uni_stable(x, a=5, b= 1):
    return b*(a - x)

# bi-stable cubic function 
def cubic_bi_stable(x, r1=4,r2=5, r3=6):
    return -1*(x-r1)*(x-r2)*(x-r3)

# straight line 
def straight_line(t, A, B):
    x1 = A[0]
    x2 = B[0]
    y1 = A[1]
    y2 = B[1]
    xt = (x2-x1)*t+x1
    yt = (y2-y1)*t+y1
    return np.transpose(np.array([xt, yt]))
