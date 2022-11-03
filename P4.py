import numpy as np

def f_x(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    f = x1**2 + x2**2 + x3**2
    return f

def h_x(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    h1 = x1**2/4 + x2**2/5 + x3**2/25
    h2 = x1 + x2 -x3
    h = np.array([[h1], [h2]])
    return h

def df_dd(d):
    x3 = d[0]
    df_d = np.array([2*x3])
    return df_d

def df_ds(s):
    x1 = s[0]
    x2 = s[1]
    df_s = np.array([2*x1, 2*x2])
    return df_s

def dh_dd(d):
    x3 = d[0]
    dh_d = np.array([[2*x3/25], [-1]])
    return dh_d

def dh_ds(s):
    x1 = float(s[0])
    x2 = float(s[1])
    dh_s = np.array([[x1/2, 2*x2/5], [1, 1]])
    return dh_s

def phi(alpha, t, d, s):
    x = np.array([s[0], s[1], d[0]])
    f = f_x(x)
    phi = f - alpha*t*np.linalg.norm(df_dd(d))
    return phi
def Inexact_Line_Search(d, s, max_iter):
    iter = 0
    alpha = 1
    b = 0.5
    t = 0.3

    d_i = d - alpha*df_dd(d)
    s_i = s + alpha*np.transpose(np.linalg.inv(dh_ds(s))@dh_dd(d)@np.transpose(df_dd(d)))
    XN = np.array([s_i[0], s_i[1], d_i[0]])
    f_alpha = f_x(XN)
    phi_alpha = phi(alpha, t, d, s)

    while f_alpha > phi_alpha and iter <= max_iter:
        if iter == max_iter:
            print('Max line search iterations hit')

        alpha = b * alpha # reduce alpha & test again

        d_i = d_i - alpha * df_dd(d_i)
        s_i = s_i + alpha * np.transpose(np.linalg.inv(dh_ds(s_i)) @ dh_dd(d_i) @ np.transpose(df_dd(d_i)))
        XN = np.array([s_i[0], s_i[1], d_i[0]])
        f_alpha = f_x(XN)
        phi_alpha = phi(alpha, t, d, s)

        iter += 1 # increment iteration number

    return alpha

def solve(d, s, max_j):
    h_p1 = h_x(np.array([s[0], s[1], d[0]]))
    j = 0

    while np.linalg.norm(h_p1) > eps and j <= max_j:
        if j == max_j:
            print('Max Newton-Ralphson iterations hit')

        q = np.linalg.inv(dh_ds(s))@h_x(np.array([float(s[0]), float(s[1]), float(d[0])]))
        s_p1_1 = s[0] - q[0]
        s_p1_2 = s[1] - q[1]
        s_p1 = np.array([float(s_p1_1), float(s_p1_2)])
        h_p1 = h_x(np.array([float(s[0]), float(s[1]), float(d[0])])) + dh_ds(s_o)@np.transpose(s_p1 - s)
        s = s_p1

        j += 1

    return s


''' GENERALIZED REDUCED GRADIENT ALGORITHM '''
# 1) Define state and decision variables:
# let s = [x1, x2]
# let d = [x3]

# 2) Initialize x0 & set other params:
x0 = np.array([-10*np.sqrt(61)/61, -10*np.sqrt(61)/61, 20*np.sqrt(61)/61]) # satisfies h(x0) = 0
s = [x0[0], x0[1]] # grab state vars
d = [x0[2]] # grab descision var(s)

eps = 1e-3

k = 0 # reduced gradient iteration number
iter = 0 # line search iteration number
max_k = 100 # reduced gradient max iterations
max_iter = 100 # line search max iterations
max_j = 100 # Newton-Ralphson max iterations

# 3) Calculate initial reduced gradient:
df_dd_kp1 = df_dd(d) - df_ds(s) @ np.linalg.inv(dh_ds(s)) @ dh_dd(d)

#4) Calculate this while loop:
while np.linalg.norm(df_dd_kp1) > eps and k <= max_k:
    if k == max_k:
        print('Max gradient descent iterations hit')
    alpha = Inexact_Line_Search(d, s, max_iter)
    d = d - alpha * df_dd(d)
    s_o = s + alpha * np.transpose(np.linalg.inv(dh_ds(s)) @ dh_dd(d) @ np.transpose(df_dd(d)))
    s = solve(d, s_o, max_j)
    df_dd_kp1 = df_dd(d) - df_ds(s) @ np.linalg.inv(dh_ds(s)) @ dh_dd(d)

    k += 1


#5) Extract solution:
print('x1 =', s[0], '\n x2 =', s[1], '\n x3 =', d[0])





