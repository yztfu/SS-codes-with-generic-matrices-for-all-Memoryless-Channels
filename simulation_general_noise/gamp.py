import numpy as np
# import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import special
from scipy import stats
from scipy import io
from scipy.interpolate import interp1d


def sparc_transforms_dct(L, B, M):
    N = L * B
    index_dct = np.random.choice(N, size=(M, ), replace=False)
    phase = np.random.choice([-1, 1], size=(N, ))

    def Ab(b):
        b = b.reshape(-1)
        z = fftpack.dct(b, norm='ortho')
        z = phase * z
        z = fftpack.idct(z, norm='ortho')
        return z[index_dct].reshape(-1, 1)

    def Az(z):
        z = z.reshape(-1)
        z_f = np.zeros((N, ))
        z_f[index_dct] = z
        b = fftpack.dct(z_f, n=N, norm='ortho')
        b = np.conj(phase) * b
        b = fftpack.idct(b, norm='ortho')
        return b.reshape(-1, 1)

    def Ab_full(b):
        b = b.reshape(-1)
        z = fftpack.dct(b, norm='ortho')
        z = phase * z
        z = fftpack.idct(z, norm='ortho')
        return z.reshape(-1, 1)

    def Az_full(z):
        z = z.reshape(-1)
        b = fftpack.dct(z, norm='ortho')
        b = np.conj(phase) * b
        b = fftpack.idct(b, norm='ortho')
        return b.reshape(-1, 1)

    return Ab, Az, Ab_full, Az_full, index_dct


def sparc_transforms_row(L, B, M):  # generate a M*N random row-orthogonal matrix
    N = L * B
    A = np.random.normal(0, 1, (M, N))  # generate a Gaussian matrix randomly
    U, S, V = np.linalg.svd(A, True, True)  # SVD of matrix A to get Haar matrix U and V
    S_new = np.hstack((np.identity(M), np.zeros((M, N - M))))
    S_new_sup = np.hstack((np.zeros((N - M, M)), np.identity(N - M)))
    A_new = U @ S_new @ V
    A_new_sup = S_new_sup @ V
    A_new_full = np.vstack((A_new, A_new_sup))
    index_gro = np.array(range(M)).reshape(-1)

    def Ab_full(b):
        z = A_new_full @ b
        return z

    def Az_full(z):
        b = A_new_full.T @ z
        return b

    return Ab_full, Az_full, index_gro


def sparc_transforms_gauss(L, B, M):
    N = L * B
    A = np.sqrt(1 / N) * np.random.normal(0, 1, (M, N))

    def Ab(b):
        z = np.dot(A, b).reshape(-1, 1)
        return z

    def Az(z):
        b = np.dot(A.T, z).reshape(-1, 1)
        return b

    return Ab, Az


def signal_awgnc(var_noise, z_0):
    M = np.shape(z_0)[0]
    noise = np.sqrt(var_noise) * np.random.randn(M, 1)
    y = z_0 + noise
    return y


def signal_bec(epsilon, z_0):
    M = np.shape(z_0)[0]
    b_0 = np.random.choice([1, 0], size=(M, 1), replace=True, p=[1 - epsilon, epsilon])
    y = b_0 * np.sign(z_0)
    return y


def signal_bsc(epsilon, z_0):
    M = np.shape(z_0)[0]
    b_0 = np.random.choice([1, -1], size=(M, 1), replace=True, p=[1 - epsilon, epsilon])
    y = b_0 * np.sign(z_0)
    return y


def signal_zc(epsilon, z_0):
    M = np.shape(z_0)[0]
    b_0 = np.random.choice([1, -1], size=(M, 1), replace=True, p=[1 - epsilon, epsilon])
    y = b_0 * np.where(z_0 < 0, -1, 0) + np.where(z_0 > 0, 1, 0)
    return y


def signal_clip(var_noise, z_0, clip):
    M = np.shape(z_0)[0]
    noise = np.sqrt(var_noise) * np.random.randn(M, 1)
    z_0_clip = np.maximum(- clip, np.minimum(clip, z_0))
    y = z_0_clip + noise
    return y


def APP_mean_var_dot_awgnc(y_in, sigma_in, mean_pri, var_pri):
    mean_post = mean_pri + var_pri / (var_pri + sigma_in) * (y_in - mean_pri)
    var_post = (var_pri * sigma_in) / (var_pri + sigma_in)
    return mean_post, var_post


def APP_mean_var_dot_bec(y_in, epsilon, mean_pri, var_pri):

    h_pos = (1 - epsilon) * np.array(y_in == -1).astype(float)
    h_neg = (1 - epsilon) * np.array(y_in == 1).astype(float)
    h_0 = 2 * epsilon * np.array(y_in == 0).astype(float)
    mean_pri_scal = mean_pri / np.sqrt(2 * var_pri)
    k = np.sqrt(2 * var_pri / np.pi) * np.exp(- mean_pri_scal ** 2) + mean_pri * special.erf(mean_pri_scal)
    k_prime = k * mean_pri + var_pri * special.erf(mean_pri_scal)

    Z = h_pos * special.erfc(mean_pri_scal) + h_neg * (1 + special.erf(mean_pri_scal)) + h_0
    Z = np.maximum(1e-12, Z)

    integral_x = h_pos * (mean_pri - k) + h_neg * (mean_pri + k) + h_0 * mean_pri

    integral_x2 = h_pos * (mean_pri ** 2 + var_pri - k_prime) \
        + h_neg * (mean_pri ** 2 + var_pri + k_prime) + h_0 * (mean_pri ** 2 + var_pri)

    mean_post = integral_x / Z
    var_post = integral_x2 / Z - mean_post ** 2
    var_post_res = float(np.mean(var_post))

    return mean_post, var_post_res


def APP_mean_var_dot_bsc(y_in, epsilon, mean_pri, var_pri):

    v_pos = (1 - epsilon) * np.array(y_in == -1).astype(float) + epsilon * np.array(y_in == 1).astype(float)
    v_neg = (1 - epsilon) * np.array(y_in == 1).astype(float) + epsilon * np.array(y_in == -1).astype(float)
    mean_pri_scal = mean_pri / np.sqrt(2 * var_pri)
    k = np.sqrt(2 * var_pri / np.pi) * np.exp(- mean_pri_scal ** 2) + mean_pri * special.erf(mean_pri_scal)
    k_prime = k * mean_pri + var_pri * special.erf(mean_pri_scal)

    Z = v_pos * special.erfc(mean_pri_scal) + v_neg * (1 + special.erf(mean_pri_scal))
    Z = np.maximum(1e-12, Z)

    integral_x = v_pos * (mean_pri - k) + v_neg * (mean_pri + k)

    integral_x2 = v_pos * (mean_pri ** 2 + var_pri - k_prime) + v_neg * (mean_pri ** 2 + var_pri + k_prime)

    mean_post = integral_x / Z
    var_post = integral_x2 / Z - mean_post ** 2
    var_post_res = float(np.mean(var_post))

    return mean_post, var_post_res


def APP_mean_var_dot_zc(y_in, epsilon, mean_pri, var_pri):

    v_pos = (1 - epsilon) * np.array(y_in == -1).astype(float) + epsilon * np.array(y_in == 1).astype(float)
    delta_y1 = np.array(y_in == 1).astype(float)
    mean_pri_scal = mean_pri / np.sqrt(2 * var_pri)
    k = np.sqrt(2 * var_pri / np.pi) * np.exp(- mean_pri_scal ** 2) + mean_pri * special.erf(mean_pri_scal)
    k_prime = k * mean_pri + var_pri * special.erf(mean_pri_scal)

    Z = v_pos * special.erfc(mean_pri_scal) + delta_y1 * (1 + special.erf(mean_pri_scal))
    Z = np.maximum(1e-12, Z)

    integral_x = v_pos * (mean_pri - k) + delta_y1 * (mean_pri + k)

    integral_x2 = v_pos * (mean_pri ** 2 + var_pri - k_prime) + delta_y1 * (mean_pri ** 2 + var_pri + k_prime)

    mean_post = integral_x / Z
    var_post = integral_x2 / Z - mean_post ** 2
    var_post_res = float(np.mean(var_post))

    return mean_post, var_post_res


def g_in(L, B, beta_PRI, Var_PRI):
    N = L * B
    rt_n_Pl = np.sqrt(B).repeat(N).reshape(-1, 1)
    u = beta_PRI * rt_n_Pl / Var_PRI
    max_u = u.reshape(L, B).max(axis=1).repeat(B).reshape(-1, 1)
    exps = np.exp(u - max_u)
    sums = exps.reshape(L, B).sum(axis=1).repeat(B).reshape(-1, 1)
    beta_POST = (rt_n_Pl * exps / sums).reshape(-1, 1)
    Var_POST = (rt_n_Pl ** 2 * (exps / sums) * (1 - exps / sums)).reshape(-1, 1)
    Var_POST_res = float(np.mean(Var_POST))
    return beta_POST, Var_POST_res


def Initialization(case, L, B, snr, epsilon, R):
    if case == 'awgnc':
        # L, B = 2 ** 12, 4
        # SNRdB_channel = 10.0
        # SNR_channel = np.power(10, SNRdB_channel / 10)
        # SNR_channel = 15.0
        # SNRdb_bool = False
        P = 1.0
        var_noise = P / snr
        # C = 0.5 * np.log2(1 + SNR_Channel)
        # Pl = pa_average(L, P)
        # R = 1.7
        N = L * B
        M = int(L * np.log2(B) / R)
        return var_noise, N, M, P
    elif case == 'bec':
        # L, B = 2 ** 10, 4
        # need change
        # epsilon = 0
        # C = 0.5 * np.log2(1 + SNR_Channel)
        # Pl = pa_average(L, P)
        # R = 0.7
        N = L * B
        M = int(L * np.log2(B) / R)
        return epsilon, N, M
    elif case == 'bsc':
        # L, B = 2 ** 10, 4
        # need change
        # epsilon = 0.1
        # C = 0.5 * np.log2(1 + SNR_Channel)
        # Pl = pa_average(L, P)
        # R = 0.55
        N = L * B
        M = int(L * np.log2(B) / R)
        return epsilon, N, M
    elif case == 'zc':
        # L, B = 2 ** 10, 4
        # need change
        # epsilon = 0.1
        # C = 0.5 * np.log2(1 + SNR_Channel)
        # Pl = pa_average(L, P)
        # R = 0.55
        N = L * B
        M = int(L * np.log2(B) / R)
        return epsilon, N, M
    elif case == 'clip':
        # L, B = 2 ** 10, 16
        # SNRdB_channel = 3.0
        # SNR_channel = np.power(10, SNRdB_channel / 10)
        # SNR_channel = 15.0
        # SNRdB_bool = True
        clipdB = 0
        clip = np.power(10, clipdB / 20)
        clipZ2 = clip ** 2 * (stats.norm.cdf(-clip) + 1 - stats.norm.cdf(clip)) \
            + stats.norm.cdf(clip) - clip * stats.norm.pdf(clip) \
            - (stats.norm.cdf(-clip) + clip * stats.norm.pdf(-clip))
        var_noise = clipZ2 / snr
        # C = 0.5 * np.log2(1 + SNR_Channel)
        # Pl = pa_average(L, P)
        # R = 1.0
        N = L * B
        M = int(L * np.log2(B) / R)
        return var_noise, N, M, clip, clipdB, clipZ2


def experiments(L, B, snr, epsilon, R, NSIM, Ite_Max):
    # Initialization
    noise_type = ['awgnc', 'bec', 'bsc', 'zc', 'clip']
    # [var_noise, N, M, _] = Initialization(noise_type[0], L, B, snr, epsilon, R)
    [epsilon, N, M] = Initialization(noise_type[1], L, B, snr, epsilon, R)
    # [epsilon, N, M] = Initialization(noise_type[2], L, B, snr, epsilon, R)
    # [epsilon, N, M] = Initialization(noise_type[3], L, B, snr, epsilon, R)
    # [var_noise, N, M, clip, _, _] = Initialization(noise_type[3], L, B, snr, epsilon, R)

    # Trial Initialization
    # NSIM = 10  # The number of trials
    # Ite_Max = 50  # The number of iterations per trial

    SER_AMP = np.zeros(Ite_Max)
    # SER_AMP_se = np.zeros(Ite_Max)
    MSE_AMP_aid = np.zeros(Ite_Max)
    MSE_AMP_cal = np.zeros(Ite_Max)
    MSE_AMP_y = np.zeros(Ite_Max)

    # ------------------------------Simulation------------------------------

    for nsim in range(NSIM):
        print("R = %.2f, nsim = %d of %d" % (R, nsim + 1, NSIM))
        # Generate random message in [0..B)^L
        tx_message = np.random.randint(0, B, L)
        tx_message.tolist()

        # Generate our transmitted signal X
        x_0 = np.zeros((N, 1))
        for l in range(L):
            x_0[l * B + tx_message[l], 0] = np.sqrt(B)

        # Generate the SPARC transform functions A.beta and A'.z
        [Ab, Az] = sparc_transforms_gauss(L, B, M)

        # Generate random channel noise and then received signal y
        z_0 = Ab(x_0)
        # y = signal_awgnc(var_noise, z_0)
        y = signal_bec(epsilon, z_0)
        # y = signal_bsc(epsilon, z_0)
        # y = signal_zc(epsilon, z_0)
        # y = signal_clip(var_noise, z_0, clip)

        # Initialization
        v_x = 1
        x_hat = np.zeros((N, 1))
        s_hat = np.zeros((M, 1))

        for it in range(Ite_Max):
            # print("nsim = %d of %d, it = %d of %d" % (nsim + 1, NSIM, it + 1, Ite_Max))

            v_p = v_x
            p_hat = Ab(x_hat) - v_p * s_hat

            # [z_hat, v_z] = APP_mean_var_dot_awgnc(y, var_noise, p_hat, v_p)
            [z_hat, v_z] = APP_mean_var_dot_bec(y, epsilon, p_hat, v_p)
            # [z_hat, v_z] = APP_mean_var_dot_bsc(y, epsilon, p_hat, v_p)
            # [z_hat, v_z] = APP_mean_var_dot_zc(y, epsilon, p_hat, v_p)
            # [z_hat, v_z] = APP_mean_var_dot_clip(y, var_noise, p_hat, v_p, clip)
            s_hat = (z_hat - p_hat) / v_p
            v_s = (v_p - v_z) / v_p ** 2

            rho_tmp = (M / N) * v_s
            v_r = 1 / rho_tmp
            r_hat = x_hat + v_r * Az(s_hat)

            [x_hat, v_x] = g_in(L, B, r_hat, v_r)

            rx_message = []
            for l in range(L):
                idx = np.argmax(x_hat[l * B: (l + 1) * B])
                rx_message.append(idx)
            SER_AMP[it] += 1 - np.sum(np.array(rx_message) == np.array(tx_message)) / L

            MSE_AMP_aid[it] += max(1e-6, np.sum((x_hat - x_0) ** 2) / N)
            MSE_AMP_cal[it] += max(1e-6, v_x)
            MSE_AMP_y[it] += max(1e-6, np.sum((Ab(x_hat) - Ab(x_0)) ** 2) / M)

            if v_x <= 1e-6:
                break

    SER_AMP /= NSIM
    MSE_AMP_aid /= NSIM
    MSE_AMP_cal /= NSIM
    MSE_AMP_y /= NSIM

    print("aid = %.3e, cal = %.3e, mse_y = %.3e" %
          (MSE_AMP_aid[-1], MSE_AMP_cal[-1], MSE_AMP_y[-1]))

    res = {'L': L, 'B': B, 'M': M, 'N': N, 'epsilon': epsilon, 'NSIM': NSIM, 'Ite_Max': Ite_Max,
           'MSE_AMP_aid': MSE_AMP_aid, 'MSE_AMP_cal': MSE_AMP_cal, 'MSE_AMP_y': MSE_AMP_y, 'SER_AMP': SER_AMP}

    io.savemat('/home/tfu/Documents/python/AMP/Conference2/results/gamp_B%d_R%.2f.mat' 
               % (B, R), res)


def main():
    # Initialization
    R_full = [0.70,]
    L, B = 2 ** 12, 4
    snr = 15.0
    epsilon = 0
    NSIM = 1      # The number of trials
    Ite_Max = 50   # The number of iterations per trial

    len_R = len(R_full)
    for r in range(len_R):
        experiments(L, B, snr, epsilon, R_full[r], NSIM, Ite_Max)


if __name__ == '__main__':
    main()
