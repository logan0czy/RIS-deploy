import numpy as np
import matplotlib.pyplot as plt


bs_loc = np.array([0, 0, 30])
radius = 10
user_center = np.array([0, 50, 0])
rect_length, rect_width = 40, 40
ris_center = np.array([-15, 50, 30])
power = 10**(20 / 10)
path_loss0 = 10**(-30 / 10)
beta_h, beta_f, beta_g = path_loss0, path_loss0, path_loss0
gamma_h, gamma_f, gamma_g = 3, 2, 2
sigma2_h, sigma2_f, sigma2_g = 1, 1, 1
rice_f, rice_g = 10, 10
noise = 10**(-110 / 10)

# constants
const1 = power*beta_h*sigma2_h / noise
const2 = power*beta_f*beta_g*rice_f*rice_g / (noise*(rice_f+1)*(rice_g+1))
const3 = 1*(rice_g*sigma2_f + rice_f*sigma2_g + sigma2_f*sigma2_g) / (rice_f*rice_g)

def gen_user_loc(nums=10, max_radius=radius, seed=None):
    """
    Return:
        (nums, 3) array
    """
    if seed:
        np.random.seed(seed)

    user_locs = np.zeros((nums, 3))
    radius = np.random.uniform(0, max_radius, nums)
    theta = np.random.uniform(0, 2*np.pi, nums)
    user_locs[:, 0] = radius * np.sin(theta)
    user_locs[:, 1] = radius * np.cos(theta)
    user_locs = user_locs + user_center[np.newaxis, :]
    return user_locs

def log_real_part(user_loc, ris_loc, coeff):
    const4 = np.prod(ris_dims) * const3
    abs2_term = np.abs(np.sum(channel_gH_bar(user_loc, ris_loc)*
        channel_f_bar(ris_loc)*coeff))**2
    res = 1 + const1*user_bs_dist(user_loc)**(-gamma_h) +\
        const2*ris_bs_dist(ris_loc)**(-gamma_f)*user_ris_dist(user_loc, ris_loc)**(-gamma_g)*\
            (const4 + abs2_term)
    return res

def cap_ub(ris_loc, user_slots, ris_coeff):
    """Capacity upper bound.

    Args:
        user_slots: (nums, )
        ris_coeff: (nums, ris_dim_flat)
    """
    acc = 0
    for user_id, user_loc in enumerate(user_locs):
        acc += user_slots[user_id] * \
            np.log2(log_real_part(user_loc, ris_loc, ris_coeff[user_id]))
    return acc

def cap(ris_loc, user_slots, ris_coeff):
    """Capacity.

    Args:
        user_slots: (nums, )
        ris_coeff: (nums, ris_dim_flat)
    """
    channel_f = np.sqrt(beta_f*ris_bs_dist(ris_loc)**(-gamma_f)) * \
        (np.sqrt(rice_f/(rice_f+1))*channel_f_bar(ris_loc) + \
            np.sqrt(1/(rice_f+1))*np.sqrt(sigma2_f/2)*\
                (np.random.randn(np.prod(ris_dims))+1j*np.random.randn(np.prod(ris_dims)))
        )
    acc = 0
    for user_id in range(user_locs.shape[0]):
        channel_h = np.sqrt(beta_h*user_bs_dist(user_locs[user_id])**(-gamma_h)) \
            * np.sqrt(sigma2_h/2) * (np.random.randn() + 1j*np.random.randn())
        channel_gH = np.sqrt(beta_g*user_ris_dist(user_locs[user_id], ris_loc)**(-gamma_g)) * \
            (np.sqrt(rice_g/(rice_g+1))*channel_gH_bar(user_locs[user_id], ris_loc) + \
                np.sqrt(1/(rice_g+1))*np.sqrt(sigma2_g/2)*\
                    (np.random.randn(np.prod(ris_dims))+1j*np.random.randn(np.prod(ris_dims)))
            )
        acc = acc + user_slots[user_id]*np.log2(1 + power/noise*\
            np.abs(channel_h + np.sum(channel_gH*ris_coeff[user_id]*channel_f))**2)
    return acc

def user_bs_dist(user_loc):
    return np.linalg.norm(user_loc - bs_loc)

def ris_bs_dist(ris_loc):
    return np.linalg.norm(ris_loc - bs_loc)

def user_ris_dist(user_loc, ris_loc):
    return np.linalg.norm(user_loc - ris_loc)

def channel_gH_bar(user_loc, ris_loc):
    """
    Return:
        gH_bar: (ris_dim_flat, )
    """
    gH_bar_power = np.zeros(ris_dims)
    for hori_id in range(ris_dims[0]):
        for vert_id in range(ris_dims[1]):
            gH_bar_power[hori_id, vert_id] = np.pi * (hori_id*(user_loc[1]-ris_loc[1]) 
                - vert_id*ris_loc[2]) / user_ris_dist(user_loc, ris_loc)
    gH_bar = np.exp(1j * gH_bar_power)
    return gH_bar.reshape(-1)

def channel_f_bar(ris_loc):
    """
    Return:
        f_bar: (ris_dim_flat, )
    """
    f_bar_power = np.zeros(ris_dims)
    for hori_id in range(ris_dims[0]):
        for vert_id in range(ris_dims[1]):
            f_bar_power[hori_id, vert_id] = -np.pi * (hori_id*ris_loc[1] 
                + vert_id*(ris_loc[2]-bs_loc[2])) / ris_bs_dist(ris_loc)
    f_bar = np.exp(1j * f_bar_power)
    return f_bar.reshape(-1)

def backtrack(func, deriv, x, direct):
    """Backtracking Line Search.

    Args:
        func: a function
    
    Return:
        alpha: step
    """
    alpha = 1
    rho, zeta = 0.9, 10**(-4)
    
    while func(x+alpha*direct) < func(x)+zeta*alpha*np.sum(deriv*direct):
        alpha *= rho
    return alpha

def proj_ris_loc(ris_loc):
    if abs(ris_loc[1]-ris_center[1])>rect_length/2:
        ris_loc[1] = ris_center[1] + np.sign(ris_loc[1]-ris_center[1])*rect_length/2
    if abs(ris_loc[2]-ris_center[2])>rect_width/2:
        ris_loc[2] = ris_center[2] + np.sign(ris_loc[2]-ris_center[2])*rect_width/2
    return ris_loc

def proj_slots(user_slots):
    for idx, t in enumerate(user_slots):
        if t<0:
            user_slots[idx] = 0
    if np.sum(user_slots)>1:
        user_slots /= np.sum(user_slots)
    return user_slots

def proj_ris_coeff(coeff):
    if np.amax(np.abs(coeff))>1:
        coeff /= np.amax(np.abs(coeff))
    return coeff

def cyclic_descent(adaptive=True):
    """Coordinate descent algoritm by adaptive steps.
    """
    lr, tole, smooth = 0.01, 10**(-2), 10**(-8)

    ris_loc = ris_center
    user_slots = np.ones(user_locs.shape[0])/user_locs.shape[0]
    ris_coeff = np.exp(1j*np.zeros((user_locs.shape[0], np.prod(ris_dims))))

    sum_gd2_loc = np.zeros(3)
    sum_gd2_slots = np.zeros(user_locs.shape[0])
    sum_gd2_coeff = [np.zeros(np.prod(ris_dims)) for _ in range(user_locs.shape[0])]
    
    print("solver start...")
    t = 1
    while True:
        # ris location
        deriv = deriv_ris_loc(ris_loc, ris_coeff, user_slots)
        sum_gd2_loc = 0.9*sum_gd2_loc + 0.1*deriv**2
        if adaptive:
            direct = proj_ris_loc(ris_loc + lr/np.sqrt(sum_gd2_loc+smooth)*deriv) - ris_loc
        else:
            direct = proj_ris_loc(ris_loc + deriv) - ris_loc
        ris_loc_next = ris_loc + direct*backtrack(lambda x: cap_ub(x, user_slots, ris_coeff), 
            deriv, ris_loc, direct)

        # slots
        deriv = deriv_slots(ris_loc_next, ris_coeff, user_slots)
        sum_gd2_slots = 0.9*sum_gd2_slots + 0.1*deriv**2
        if adaptive:
            direct = proj_slots(user_slots + lr/np.sqrt(sum_gd2_slots+smooth)*deriv) - user_slots
        else:
            direct = proj_slots(user_slots + deriv) - user_slots

        user_slots_next = user_slots + direct*backtrack(lambda x: cap_ub(ris_loc_next, x, ris_coeff), 
            deriv, user_slots, direct)

        # coeff
        ris_coeff_next = np.zeros(ris_coeff.shape)
        for idx, user_loc in enumerate(user_locs):
            deriv = deriv_ris_coeff(user_loc, ris_loc_next, ris_coeff[idx], user_slots_next[idx])
            sum_gd2_coeff[idx] = 0.9*sum_gd2_coeff[idx] + 0.1*deriv**2
            if adaptive:
                direct = proj_ris_coeff(ris_coeff[idx] + lr/np.sqrt(sum_gd2_coeff[idx]+smooth)*deriv) - ris_coeff[idx]
            else:
                direct = proj_ris_coeff(ris_coeff[idx] + deriv) - ris_coeff[idx]
            ris_coeff_next[idx] = ris_coeff[idx] + \
                direct*backtrack(lambda x: cap_ub(ris_loc_next, user_slots_next, np.stack((ris_coeff_next[:idx], x, ris_coeff[idx+1:]))),
                    deriv, ris_coeff[idx], direct)
        
        t += 1
        var = np.concatenate((ris_loc, user_slots, ris_coeff.reshape(-1)))
        var_next = np.concatenate((ris_loc_next, user_slots_next, ris_coeff_next.reshape(-1)))
        if np.linalg.norm(var_next - var) <= tole:
            ris_loc = ris_loc_next
            user_slots = user_slots_next
            ris_coeff = ris_coeff_next
            break

        ris_loc = ris_loc_next
        user_slots = user_slots_next
        ris_coeff = ris_coeff_next
        if t%5==0:
            print(f"step {t}, current capacity upper bound {cap_ub(ris_loc, user_slots, ris_coeff)}")
    return ris_loc, user_slots, ris_coeff

def deriv_ris_loc(ris_loc, ris_coeff, user_slots):
    """Derivation of capacity upper bound to ris location.
    """

    def ris_bs_deriv(ris_loc):
        dist = ris_bs_dist(ris_loc)
        return np.array([0, ris_loc[1]/dist, (ris_loc[2]-bs_loc[2])/dist])

    def user_ris_deriv(user_loc, ris_loc):
        dist = user_ris_dist(user_loc, ris_loc)
        return np.array([0, -(user_loc[1]-ris_loc[1])/dist, ris_loc[2]/dist])
    
    def delta_f_deriv(ris_loc):
        """
        Return:
            (deriv to coordinate y, deriv to coordinate z)
        """
        deriv_y = np.zeros(ris_dims)
        deriv_z = np.zeros(ris_dims)
        for hori_id in range(ris_dims[0]):
            for vert_id in range(ris_dims[1]):
                molecular = hori_id*ris_loc[1] + vert_id*(ris_loc[2]-bs_loc[2])
                deriv_y[hori_id, vert_id] = -np.pi / ris_bs_dist(ris_loc)**2 *\
                    (hori_id*ris_bs_dist(ris_loc) - molecular*ris_bs_deriv(ris_loc)[1])
                deriv_z[hori_id, vert_id] = -np.pi / ris_bs_dist(ris_loc)**2 *\
                    (vert_id*ris_bs_dist(ris_loc) - molecular*ris_bs_deriv(ris_loc)[2])
        return np.stack((deriv_y.reshape(-1), deriv_z.reshape(-1)))

    def delta_g_deriv(user_loc, ris_loc):
        """
        Return:
            (deriv to coordinate y, deriv to coordinate z)
        """
        deriv_y = np.zeros(ris_dims)
        deriv_z = np.zeros(ris_dims)
        for hori_id in range(ris_dims[0]):
            for vert_id in range(ris_dims[1]):
                molecular = hori_id*(user_loc[1]-ris_loc[1]) - vert_id*ris_loc[2]
                deriv_y[hori_id, vert_id] = np.pi / user_ris_dist(user_loc, ris_loc)**2 *\
                    (-hori_id*user_ris_dist(user_loc, ris_loc) - molecular*user_ris_deriv(user_loc, ris_loc)[1])
                deriv_z[hori_id, vert_id] = np.pi / user_ris_dist(user_loc, ris_loc)**2 *\
                    (-vert_id*user_ris_dist(user_loc, ris_loc) - molecular*user_ris_deriv(user_loc, ris_loc)[2] )
        return np.stack((deriv_y.reshape(-1), deriv_z.reshape(-1)))

    def abs2_deriv(user_loc, ris_loc, coeff):
        g_f_deriv = (channel_gH_bar(user_loc, ris_loc)*channel_f_bar(ris_loc)*1j)[np.newaxis, :] *\
            (delta_f_deriv(ris_loc) + delta_g_deriv(user_loc, ris_loc))
        term = np.sum(channel_gH_bar(user_loc, ris_loc)*
            channel_f_bar(ris_loc)*coeff)  ## scaler
        term_deriv = np.sum(coeff[np.newaxis, :]*g_f_deriv, axis=1)
        res = np.conj(term_deriv)*term + np.conj(term)*term_deriv
        return np.array([0, res[0], res[1]])

    const4 = np.prod(ris_dims)*const3
    res = np.zeros(3)
    for user_id, user_loc in enumerate(user_locs):
        abs2_term = np.abs(np.sum(channel_gH_bar(user_loc, ris_loc)*
            channel_f_bar(ris_loc)*ris_coeff[user_id]))**2
        res += user_slots[user_id] / (np.log(2)*log_real_part(user_loc, ris_loc, ris_coeff[user_id])) * const2\
            (
            (-gamma_f)*ris_bs_dist(ris_loc)**(-gamma_f-1)*ris_bs_deriv(ris_loc)*user_ris_dist(user_loc, ris_loc)**(-gamma_g)*(const4+abs2_term) +
            (-gamma_g)*user_ris_dist(user_loc, ris_loc)**(-gamma_g-1)*user_ris_deriv(user_loc, ris_loc)*ris_bs_dist(ris_loc)**(-gamma_f)*(const4 + abs2_term) +
            ris_bs_dist(ris_loc)**(-gamma_f)*user_ris_dist(user_loc, ris_loc)**(-gamma_g)*abs2_deriv(user_loc, ris_loc, ris_coeff[user_id])
            )
    return res

def deriv_slots(ris_loc, ris_coeff, user_slots):
    res = np.zeros(user_locs.shape[0])
    for idx, user_loc in enumerate(user_locs):
        res[idx] = np.log2(log_real_part(user_loc, ris_loc, ris_coeff[idx]))
    return res

def deriv_ris_coeff(user_loc, ris_loc, coeff, slot):
    """Derivation to one of coefficient w.r.t a user"""
    term = channel_gH_bar(user_loc, ris_loc)*channel_f_bar(ris_loc)
    res = slot / (np.log(2)*log_real_part(user_loc, ris_loc, coeff)) *\
        const2*ris_bs_dist(ris_loc)**(-gamma_f)*user_ris_dist(user_loc, ris_loc)**(-gamma_g) *\
            np.conj(term)*np.sum(term*coeff)
    return res

if __name__=='__main__':
    user_locs = np.zeros((3,4))
    ris_dims = (4, 6)
    pass