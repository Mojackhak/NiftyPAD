__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import inspect

from niftypad import kt
from niftypad import kp
from niftypad import basis

# # for debugging
from numpy import savetxt
from scipy.io import savemat


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # linear models
# # # # # # # # # # # # # # # # # # # # # # # # # # # #


# srtmb_basis - srtm model for img with pre-calculated basis functions

def srtmb_basis(tac, b):
    tac[tac < 0] = 0.0

    n_beta = b['beta'].size
    ssq = np.zeros(n_beta)

    if b['w'] is None:
        b['w'] = 1

    for i in range(0, n_beta):
        theta = np.dot(b['m_w'][i * 2:i * 2 + 2, :], b['w'] * tac)
        a = np.column_stack((b['input'], b['basis'][i, :]))
        tacf = np.dot(a, theta)
        res = (tac - tacf) * b['w']
        ssq[i] = np.sum(res ** 2)
    i = np.argmin(ssq)
    theta = np.dot(b['m_w'][i * 2:i * 2 + 2, :], b['w'] * tac)
    a = np.column_stack((b['input'], b['basis'][i, :]))
    tacf = np.dot(a, theta)

    theta = np.append(theta, b['beta'][i])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    k2p = k2 / r1
    kps = {'r1': r1, 'k2': k2, 'bp': bp, 'k2p': k2p, 'tacf': tacf}
    return kps


def srtmb_basis_ppet(tac, b):
    tac[tac < 0] = 0.0

    n_beta = b['beta'].size
    ssq = np.zeros(n_beta)

    if b['w'] is None:
        b['w'] = 1

    for i in range(0, n_beta):
        a = np.column_stack((b['input'], b['basis'][i, :]))
        theta = np.linalg.inv(a.T @ a) @ a.T @ tac
        tacf = np.dot(a, theta)
        res = (tac - tacf) * b['w']
        ssq[i] = np.sum(res ** 2)
    i = np.argmin(ssq)
    a = np.column_stack((b['input'], b['basis'][i, :]))
    theta = np.linalg.inv(a.T @ a) @ a.T @ tac
    tacf = np.dot(a, theta)

    theta = np.append(theta, b['beta'][i])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    kps = {'r1': r1, 'k2': k2, 'bp': bp, 'tacf': tacf}
    return kps


def srtmb_basis_para2tac(r1, k2, bp, b):
    tacf = []
    theta = kp.srtm_kp2theta(r1, k2, bp)
    i = np.argwhere(abs(b['beta'] - theta[-1]) < 1e-10).squeeze()
    if not i == []:
        a = np.column_stack((b['input'], b['basis'][i, :]))
        tacf = np.dot(a, theta[:-1])
    kps = {'tacf': tacf}
    return kps


# srtmb - srtm model for tac, basis functions will be calculated

def srtmb(tac, dt, inputf1, beta_lim, n_beta, w):
    tac[tac < 0] = 0.0
    b = basis.make_basis(inputf1, dt, beta_lim=beta_lim, n_beta=n_beta, w=w)
    kps = srtmb_basis(tac, b)
    return kps


# srtmb_asl_basis - srtm model for tac with fixed R1 and pre-calculated basis functions

def srtmb_asl_basis(tac, b, r1):
    tac[tac < 0] = 0.0

    n_beta = b['beta'].size
    ssq = np.zeros(n_beta)

    if b['w'] is None:
        b['w'] = 1
    y = tac - r1 * b['input']
    for i in range(0, n_beta):
        theta = r1
        theta = np.append(theta, np.dot(y, b['basis'][i, :]) / np.dot(b['basis'][i, :], b['basis'][i,
                                                                                        :]))  # w won't make a difference here
        a = np.column_stack((b['input'], b['basis'][i, :]))
        tacf = np.dot(a, theta)
        res = (tac - tacf) * b['w']  # w works here
        ssq[i] = np.sum(res ** 2)
    i = np.argmin(ssq)
    theta = r1
    theta = np.append(theta, np.dot(y, b['basis'][i, :]) / np.dot(b['basis'][i, :], b['basis'][i, :]))
    a = np.column_stack((b['input'], b['basis'][i, :]))
    tacf = np.dot(a, theta)

    theta = np.append(theta, b['beta'][i])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    kps = {'r1': r1, 'k2': k2, 'bp': bp, 'tacf': tacf}
    return kps


# srtmb_asl - srtm model for tac with fixed R1, basis functions will be calculated

def srtmb_asl(tac, dt, inputf1, beta_lim, n_beta, w, r1):
    tac[tac < 0] = 0.0
    b = basis.make_basis(inputf1, dt, beta_lim=beta_lim, n_beta=n_beta, w=w)
    kps = srtmb_asl_basis(tac, b, r1)
    return kps


# srtmb_k2p_basis - srtm model for img with fixed k2p and pre-calculated basis functions

def srtmb_k2p_basis(tac, b):
    tac[tac < 0] = 0.0

    n_beta = b['beta'].size
    ssq = np.zeros(n_beta)

    if b['w'] is None:
        b['w'] = np.ones_like(tac)

    for i in range(0, n_beta):

        r1 = np.sum(b['w'] * b['basis_k2p'][i] * tac) / np.sum(b['w'] * b['basis_k2p'][i] ** 2)
        ssq[i] = np.sum(b['w'] * (tac - r1 * b['basis_k2p'][i]) ** 2)

        # print(i)
        # print(tac)
        # print(np.sum(b['w'] * b['basis_k2p'][i] * tac))
        # print(np.sum(b['w'] * b['basis_k2p'][i] ** 2))
        # print(r1)
        # print(ssq[i])
    i = np.argmin(ssq)
    r1 = np.sum(b['w'] * b['basis_k2p'][i] * tac) / np.sum(b['w'] * b['basis_k2p'][i] ** 2)
    tacf = r1 * b['basis_k2p'][i]

    theta = r1
    theta = np.append(theta, r1 * (b['k2p'] - b['beta'][i]))
    theta = np.append(theta, b['beta'][i])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    kps = {'r1': r1, 'k2': k2, 'bp': bp, 'tacf': tacf}
    return kps


def srtmb_k2p_basis_para2tac(r1, k2, bp, b):
    tacf = []
    theta = kp.srtm_kp2theta(r1, k2, bp)
    i = np.argwhere(abs(b['beta'] - theta[-1]) < 1e-10).squeeze()
    if not i == []:
        tacf = theta[0] * b['basis_k2p'][i]
    kps = {'tacf': tacf}
    return kps


# srtmb_k2p - srtm model for tac with fixed k2p, basis functions will be calculated


def srtmb_k2p(tac, dt, inputf1, beta_lim, n_beta, w, k2p):
    tac[tac < 0] = 0.0
    b = basis.make_basis(inputf1, dt, beta_lim=beta_lim, n_beta=n_beta, w=w, k2p=k2p)
    kps = srtmb_k2p_basis(tac, b)
    return kps


# # # # # # graphic models

# logan_ref - logan reference plot without fixed k2p for tac, based on eq.7 in
# "Distribution Volume Ratios Without Blood Sampling from Graphical Analysis of PET Data"

def logan_ref(tac, dt, inputf1, w, linear_phase_start_l, linear_phase_end_l, fig):
    if w is None:
        w = np.ones_like(tac)
    linear_phase_start = linear_phase_start_l
    linear_phase_end = linear_phase_end_l
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    if linear_phase_end > np.amax(dt):
        linear_phase_end = np.amax(dt)

    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        w, _ = kt.tac_dt_fill_coffee_break(w, dt, interp='zero')
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt, interp='linear')
    mft = kt.dt2mft(dt)
    # print(mft)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    # input_dt = kt.int2dt(inputf1, dt)
    inputff = interp1d(np.arange(len(inputf1)), inputf1, kind='linear', fill_value='extrapolate')
    input_dt = inputff(mft)
    input_dt = input_dt[1:]

    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    # calculate integration
    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    input_cum = np.cumsum((input_dt[:-1] + input_dt[1:]) / 2 * tdur)
    # tac_cum and input_cum are calculated in such a way to match tac

    # # # debug
    #     file_path = '/Users/Himiko/data/amsterdam_data_pib/transfer_162391_files_89f2cba1/'
    #     savetxt(file_path + 'tac_cum.csv', tac_cum)
    #     savetxt(file_path + 'input_cum.csv', input_cum)
    #
    #     tac1 = kt.interpt1(mft, tac, dt)
    #     input1 = kt.interpt1(mft, input_dt, dt)
    #     tac_cum1 = np.cumsum(tac1)
    #     input_cum1 = np.cumsum(input1)
    #     tac_cum1_mft = tac_cum1[mft.astype(int)]
    #     input_cum1_mft = input_cum1[mft.astype(int)]
    #     savetxt(file_path + 'tac_cum1_mft.csv', tac_cum1_mft)
    #     savetxt(file_path + 'input_cum1_mft.csv', input_cum1_mft)
    # # # debug

    tac = tac[1:]
    input_dt = input_dt[1:]

    # yy = np.zeros(tac.shape)
    # xx = np.zeros(tac.shape)
    # mask = tac.nonzero()
    # yy[mask] = tac_cum[mask] / tac[mask]
    # xx[mask] = input_cum[mask] / tac[mask]

    yy = tac_cum / (tac + 0.0000000000000001)  # ADDED BY MY 20210616
    xx = input_cum / (tac + 0.0000000000000001)  # ADDED BY MY 20210616

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)
    tt = tt[1:]
    # select tt for tac > 0
    tt = np.logical_and(tt, tac > 0)
    # select tt for xx < inf, yy < inf
    infinf = 1e10
    tt = np.logical_and(tt, xx < infinf)
    tt = np.logical_and(tt, yy < infinf)
    tt = np.logical_and(tt, w > 0)

    # do linear regression with selected tt
    # xx = xx[tt]
    # yy = yy[tt]
    # dvr, inter, _, _, _ = linregress(xx, yy)
    # bp = dvr - 1
    # yyf = dvr * xx + inter
    try:
        reg = LinearRegression().fit(xx[tt].reshape(-1, 1), yy[tt], sample_weight=w[tt])
        dvr = reg.coef_[0]
        bp = dvr - 1
        yyf = reg.predict(xx.reshape(-1, 1))
    except:
        bp = 0
        yyf = yy[tt] * 0



    if fig:
        plt.plot(xx, yy, '.')
        plt.plot(xx, yyf, 'r')
        plt.show()
    kps = {'bp': bp}
    return kps


# logan_ref_k2p - logan reference plot with fixed k2p for tac, based on eq.6 in
# # "Distribution Volume Ratios Without Blood Sampling from Graphical Analysis of PET Data"

def logan_ref_k2p(tac, dt, inputf1, k2p, w, linear_phase_start_l2, linear_phase_end_l2, fig):
    if w is None:
        w = np.ones_like(tac)
    linear_phase_start = linear_phase_start_l2
    linear_phase_end = linear_phase_end_l2
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    if linear_phase_end > np.amax(dt):
        linear_phase_end = np.amax(dt)
    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        w, _ = kt.tac_dt_fill_coffee_break(w, dt, interp='zero')
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt, interp='linear')
    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    # input_dt = kt.int2dt(inputf1, dt)
    inputff = interp1d(np.arange(len(inputf1)), inputf1, kind='linear', fill_value='extrapolate')
    input_dt = inputff(mft)
    input_dt = input_dt[1:]
    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    input_cum = np.cumsum((input_dt[:-1] + input_dt[1:]) / 2 * tdur)
    tac = tac[1:]
    input_dt = input_dt[1:]

    yy = tac_cum / (tac + 0.0000000000000001)
    xx = (input_cum + input_dt / k2p) / (tac + 0.0000000000000001)

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)
    tt = tt[1:]
    # select tt for tac > 0
    tt = np.logical_and(tt, tac > 0)
    # select tt for xx < inf, yy < inf
    infinf = 1e20
    tt = np.logical_and(tt, xx < infinf)
    tt = np.logical_and(tt, yy < infinf)
    tt = np.logical_and(tt, w > 0)


    # do linear regression with selected tt
    # xx = xx[tt]
    # yy = yy[tt]
    # dvr, inter, _, _, _ = linregress(xx, yy)
    # bp = dvr - 1
    # yyf = dvr * xx + inter

    reg = LinearRegression().fit(xx[tt].reshape(-1, 1), yy[tt], sample_weight=w[tt])
    dvr = reg.coef_[0]
    bp = dvr - 1
    yyf = reg.predict(xx.reshape(-1, 1))

    if fig:
        plt.plot(xx, yy, '.')
        plt.plot(xx[tt], yyf[tt], 'r')
        plt.show()
    kps = {'bp': bp}
    return kps


# mrtm - Ichise's multilinear reference tissue model

def mrtm(tac, dt, inputf1, w, linear_phase_start_m, linear_phase_end_m, fig):
    if w is None:
        w = np.ones_like(tac)
    linear_phase_start = linear_phase_start_m
    linear_phase_end = linear_phase_end_m
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    if linear_phase_end > np.amax(dt):
        linear_phase_end = np.amax(dt)
    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        w, _ = kt.tac_dt_fill_coffee_break(w, dt, interp='zero')
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt, interp='linear')
    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    # input_dt = kt.int2dt(inputf1, dt)
    inputff = interp1d(np.arange(len(inputf1)), inputf1, kind='linear', fill_value='extrapolate')
    input_dt = inputff(mft)
    input_dt = input_dt[1:]
    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    # print(tdur)
    # print(tac_cum)
    input_cum = np.cumsum((input_dt[:-1] + input_dt[1:]) / 2 * tdur)
    tac = tac[1:]
    input_dt = input_dt[1:]
    yy = tac
    xx = np.column_stack((input_cum, tac_cum, input_dt))

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)

    tt = tt[1:]

    mft = mft[1:]
    # print(xx[tt, ])
    # print(yy[tt])

    tt = np.logical_and(tt, w > 0)

    reg = LinearRegression(fit_intercept=False).fit(xx[tt, ], yy[tt], sample_weight=w[tt])
    bp = - reg.coef_[0] / reg.coef_[1] - 1
    k2p = reg.coef_[0] / reg.coef_[2]
    # for 1 TC
    r1 = reg.coef_[2]
    k2 = - reg.coef_[1]


    # # par = INVERT(XT  ##WTS##X)##XT##WTS##transpose(yfit(*))
    # reg = np.linalg.inv(xx.T @ xx) @ xx.T @ yy
    # bp = - reg[0] / reg[1] - 1
    # k2p = reg[0] / reg[2]
    #
    # # # for 1 TC
    # r1 = reg[2]
    # k2 = - reg[1]

    if np.isnan(bp):
        bp = 0
    if np.isnan(r1):
        r1 = 1.0
    if r1 > 5:
        r1 = 5
    if r1 < -5:
        r1 = -5
    if bp > 10:
        bp = 0
    if bp < -10:
        bp = 0


    # if r1 > 5:
    #     r1 = 1
    # if r1 < -5:
    #     r1 = 1
    # if bp > 5:
    #     bp = 0
    # if bp < -5:
    #     bp = 0



    # yyf = reg.predict(xx)
    if fig:
        plt.plot(mft, yy, '.')
        plt.plot(mft, yyf, 'r')
        plt.show()
    kps = {'bp': bp, 'k2p': k2p, 'r1': r1, 'k2': k2}

    return kps


# mrtm - Ichise's multilinear reference tissue model with fixed k2prime

def mrtm_k2p(tac, dt, inputf1, k2p, w, linear_phase_start_m2, linear_phase_end_m2, fig):
    if w is None:
        w = np.ones_like(tac)
    linear_phase_start = linear_phase_start_m2
    linear_phase_end = linear_phase_end_m2
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    if linear_phase_end > np.amax(dt):
        linear_phase_end = np.amax(dt)
    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        w, _ = kt.tac_dt_fill_coffee_break(w, dt, interp='zero')
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt, interp='linear')
    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    # input_dt = kt.int2dt(inputf1,dt)
    inputff = interp1d(np.arange(len(inputf1)), inputf1, kind='linear', fill_value='extrapolate')
    input_dt = inputff(mft)
    input_dt = input_dt[1:]
    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    input_cum = np.cumsum((input_dt[:-1] + input_dt[1:]) / 2 * tdur)
    tac = tac[1:]
    input_dt = input_dt[1:]

    yy = tac
    xx = np.column_stack((input_cum + 1 / k2p * input_dt, tac_cum))

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)
    tt = tt[1:]

    mft = mft[1:]
    # print(xx[tt, ])
    # print(yy[tt])
    tt = np.logical_and(tt, w > 0)

    reg = LinearRegression(fit_intercept=False).fit(xx[tt, ], yy[tt], sample_weight=w[tt])
    bp = - reg.coef_[0] / reg.coef_[1] - 1

    # for 1 TC
    k2 = -reg.coef_[1]
    r1 = reg.coef_[0] / k2p

    if np.isnan(bp):
        bp = 0
    if np.isnan(r1):
        r1 = 1.0
    if r1 > 5:
        r1 = 5
    if r1 < -5:
        r1 = -5
    if bp > 10:
        bp = 0
    if bp < -10:
        bp = 0

    yyf = reg.predict(xx)
    if fig:
        plt.plot(mft, yy, '.')
        plt.plot(mft, yyf, 'r')
        plt.show()
    kps = {'bp': bp, 'r1': r1}
    return kps


# # # # # # graphic models - PPET version

# logan_ref - logan reference plot without fixed k2p for tac, based on eq.7 in
# "Distribution Volume Ratios Without Blood Sampling from Graphical Analysis of PET Data"
#  PPET version: calculate input_dt and input_cum differently
def logan_ref_ppet(tac, dt, ref, linear_phase_start, linear_phase_end, fig):
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt)
    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    # get input_dt from ref.tac, if dt and ref.dt are the same
    if np.array_equal(dt, ref.dt):
        input_dt = ref.tac
    else:
        inputff = interp1d(kt.dt2mft(ref.dt), ref.tac, kind='linear', fill_value='extrapolate')
        input_dt = inputff(mft)
        input_dt = input_dt[1:]

    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    # calculate integration
    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    # calculate input_cum using inputf1, and only until mid frame time
    inputf1 = kt.interpt1(kt.dt2mft(ref.dt), ref.tac, dt)
    input_cum1 = np.cumsum(inputf1)
    input_cum1_mft = input_cum1[mft.astype(int)]
    input_cum = input_cum1_mft
    # tac_cum and input_cum are calculated in such a way to match tac

    tac = tac[1:]
    input_dt = input_dt[1:]
    input_cum = input_cum[1:]

    yy = tac_cum / (tac + 0.0000000000000001)  # ADDED BY MY 20210616
    xx = input_cum / (tac + 0.0000000000000001)  # ADDED BY MY 20210616

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)
    tt = tt[1:]
    # select tt for tac > 0
    tt = np.logical_and(tt, tac > 0)
    # select tt for xx < inf, yy < inf
    infinf = 1e10
    tt = np.logical_and(tt, xx < infinf)
    tt = np.logical_and(tt, yy < infinf)

    # do linear regression with selected tt
    xx = xx[tt]
    yy = yy[tt]

    dvr, inter, _, _, _ = linregress(xx, yy)
    bp = dvr - 1
    yyf = dvr * xx + inter
    if fig:
        plt.plot(xx, yy, '.')
        plt.plot(xx, yyf, 'r')
        plt.show()
    kps = {'bp': bp}
    return kps


# logan_ref_k2p - logan reference plot with fixed k2p for tac, based on eq.6 in
# # "Distribution Volume Ratios Without Blood Sampling from Graphical Analysis of PET Data"
#  PPET version: calculate input_dt and input_cum differently
def logan_ref_k2p_ppet(tac, dt, ref, k2p, w, linear_phase_start, linear_phase_end, fig):
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt)
    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    # get input_dt from ref.tac, if dt and ref.dt are the same
    if np.array_equal(dt, ref.dt):
        input_dt = ref.tac
    else:
        inputff = interp1d(kt.dt2mft(ref.dt), ref.tac, kind='linear', fill_value='extrapolate')
        input_dt = inputff(mft)
        input_dt = input_dt[1:]

    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    # calculate input_cum using inputf1, and only until mid frame time
    inputf1 = kt.interpt1(kt.dt2mft(ref.dt), ref.tac, dt)
    input_cum1 = np.cumsum(inputf1)
    input_cum1_mft = input_cum1[mft.astype(int)]
    input_cum = input_cum1_mft
    # tac_cum and input_cum are calculated in such a way to match tac
    tac = tac[1:]
    input_dt = input_dt[1:]
    input_cum = input_cum[1:]

    yy = tac_cum / (tac + 0.0000000000000001)
    xx = (input_cum + input_dt / k2p) / (tac + 0.0000000000000001)

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)
    tt = tt[1:]
    # select tt for tac > 0
    tt = np.logical_and(tt, tac > 0)
    # select tt for xx < inf, yy < inf
    infinf = 1e10
    tt = np.logical_and(tt, xx < infinf)
    tt = np.logical_and(tt, yy < infinf)

    # do linear regression with selected tt
    xx = xx[tt]
    yy = yy[tt]

    dvr, inter, _, _, _ = linregress(xx, yy)
    bp = dvr - 1
    yyf = dvr * xx + inter
    if fig:
        plt.plot(xx, yy, '.')
        plt.plot(xx[tt], yyf[tt], 'r')
        plt.show()
    kps = {'bp': bp}
    return kps


# mrtm - Ichise's multilinear reference tissue model
#  PPET version: calculate input_dt and input_cum, tac_cum differently
def mrtm_ppet(tac, dt, inputf1, w, linear_phase_start_m, linear_phase_end_m, fig):
    if w is None:
        w = np.ones_like(tac)
    linear_phase_start = linear_phase_start_m
    linear_phase_end = linear_phase_end_m
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    if linear_phase_end > np.amax(dt):
        linear_phase_end = np.amax(dt)
    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        w, _ = kt.tac_dt_fill_coffee_break(w, dt, interp='zero')
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt, interp='linear')

    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)

    # input_dt = kt.int2dt(inputf1, dt)
    inputff = interp1d(np.arange(len(inputf1)), inputf1, kind='linear', fill_value='extrapolate')
    input_dt = inputff(mft)
    input_dt = input_dt[1:]
    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    # calculate input_cum using inputf1, and only until mid frame time
    input_cum1 = np.cumsum(inputf1)
    input_cum1_mft = input_cum1[mft.astype(int)]
    input_cum = input_cum1_mft
    # tac_cum and input_cum are calculated in such a way to match tac
    tac = tac[1:]
    input_dt = input_dt[1:]
    input_cum = input_cum[1:]

    # PPET tac_cum calculation
    tac_cum = np.zeros_like(input_cum)
    mft_tac = mft[1:]
    for i in range(len(mft_tac)):
        if i == 0:
            tac_cum[i] = tac[i] * (mft_tac[i] - dt[0, i])
        else:
            tac_cum[i] = tac_cum[i-1] + tac[i-1] * (dt[1, i-1] - mft_tac[i-1]) + tac[i] * (mft_tac[i] - dt[0, i])
    # PPET tac_cum calculation

    yy = tac
    xx = np.column_stack((input_cum, tac_cum, input_dt))

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)
    tt = tt[1:]

    mft = mft[1:]

    reg = LinearRegression(fit_intercept=False).fit(xx[tt, ], yy[tt], sample_weight=w[tt])
    bp = - reg.coef_[0] / (reg.coef_[1] + 0.0000000000000001) - 1
    k2p = reg.coef_[0] / (reg.coef_[2] + 0.0000000000000001)
    # for 1 TC
    r1 = reg.coef_[2]
    k2 = - reg.coef_[1]

    if np.isnan(bp):
        bp = 0
    if np.isnan(r1):
        r1 = 1.0
    if r1 > 5:
        r1 = 5
    if r1 < -5:
        r1 = -5
    if bp > 10:
        bp = 0
    if bp < -10:
        bp = 0

    yyf = reg.predict(xx)
    if fig:
        plt.plot(mft, yy, '.')
        plt.plot(mft, yyf, 'r')
        plt.show()
    kps = {'bp': bp, 'k2p': k2p, 'r1': r1, 'k2': k2}
    return kps


# mrtm - Ichise's multilinear reference tissue model with fixed k2prime
#  PPET version: calculate input_dt and input_cum differently
def mrtm_k2p_ppet(tac, dt, ref, k2p, w, linear_phase_start, linear_phase_end, fig):
    if linear_phase_start is None:
        linear_phase_start = 0
    if linear_phase_end is None:
        linear_phase_end = np.amax(dt)
    # fill the coffee break gap
    if kt.dt_has_gaps(dt):
        tac, dt = kt.tac_dt_fill_coffee_break(tac, dt)
    mft = kt.dt2mft(dt)
    mft = np.append(0, mft)
    dt_new = np.array([mft[:-1], mft[1:]])
    tdur = kt.dt2tdur(dt_new)
    # get input_dt from ref.tac, if dt and ref.dt are the same
    if np.array_equal(dt, ref.dt):
        input_dt = ref.tac
    else:
        inputff = interp1d(kt.dt2mft(ref.dt), ref.tac, kind='linear', fill_value='extrapolate')
        input_dt = inputff(mft)
        input_dt = input_dt[1:]

    tac = np.append(0, tac)
    input_dt = np.append(0, input_dt)

    # set negative values to zero
    tac[tac < 0] = 0.0
    input_dt[input_dt < 0] = 0.0

    tac_cum = np.cumsum((tac[:-1] + tac[1:]) / 2 * tdur)
    # calculate input_cum using inputf1, and only until mid frame time
    inputf1 = kt.interpt1(kt.dt2mft(ref.dt), ref.tac, dt)
    input_cum1 = np.cumsum(inputf1)
    input_cum1_mft = input_cum1[mft.astype(int)]
    input_cum = input_cum1_mft
    # tac_cum and input_cum are calculated in such a way to match tac
    tac = tac[1:]
    input_dt = input_dt[1:]
    input_cum = input_cum[1:]

    yy = tac
    xx = np.column_stack((input_cum + 1 / k2p * input_dt, tac_cum))

    # find tt for the linear phase
    tt = np.logical_and(mft >= linear_phase_start, mft <= linear_phase_end)
    tt = tt[1:]

    mft = mft[1:]
    reg = LinearRegression(fit_intercept=False).fit(xx[tt,], yy[tt], sample_weight=w)
    bp = - reg.coef_[0] / reg.coef_[1] - 1

    # for 1 TC
    k2 = -reg.coef_[1]
    r1 = reg.coef_[0] / k2p

    if np.isnan(bp):
        bp = 0
    if np.isnan(r1):
        r1 = 1.0
    if r1 > 5:
        r1 = 5
    if r1 < -5:
        r1 = -5
    if bp > 10:
        bp = 0
    if bp < -10:
        bp = 0

    yyf = reg.predict(xx)
    if fig:
        plt.plot(mft, yy, '.')
        plt.plot(mft, yyf, 'r')
        plt.show()
    kps = {'bp': bp, 'r1': r1}
    return kps


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # non-linear models
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# srtm - srtm model for tac, with non-linear optimisation

def srtm_fun(inputf1_dt, r1, k2, bp):
    inputf1, dt = inputf1_dt
    t1 = np.arange(np.amax(dt))
    theta = kp.srtm_kp2theta(r1, k2, bp)
    tac1 = theta[0] * inputf1
    tac1 += theta[1] * np.convolve(inputf1, np.exp(-theta[2] * t1))[0:tac1.size]
    tac = kt.int2dt(tac1, dt)
    return tac


def srtm_para2tac(r1, k2, bp, inputf1_dt):
    tacf = srtm_fun(inputf1_dt, r1, k2, bp)
    kps = {'tacf': tacf}
    return kps


def srtm_fun_w(inputf1_dt_w, r1, k2, bp):
    inputf1, dt1, dt2, w = inputf1_dt_w
    dt = np.array([dt1, dt2])
    tac = srtm_fun((inputf1, dt), r1, k2, bp)
    if w is None:
        w = 1
    tac_w = tac * w
    return tac_w


def srtm(tac, dt, inputf1, w):

    if w is None:
        w = 1

    if np.isscalar(w):
        w = np.full_like(dt[0,], w)
    
    inputf1 = kt.int2dt(inputf1, dt)
    inputf1_dt = inputf1, dt
    inputf1_dt_w = inputf1, dt[0,], dt[1,], w   # must be k*M matrix      
    
    p, _ = curve_fit(srtm_fun_w, inputf1_dt_w, tac * w, p0=[1, 0.00005, 0.0], bounds=(0, [3, 1, 10]))
    r1 = p[0]
    k2 = p[1]
    bp = p[2]
    tacf = srtm_fun(inputf1_dt, r1, k2, bp)
    k2p = k2 / r1
    kps = {'r1': r1, 'k2': k2, 'bp': bp, 'k2p': k2p, 'tacf': tacf}
    return kps


# srtm_k2p - srtm model for tac with fixed k2p, with non-linear optimisation


def srtm_fun_k2p(inputf1_dt_k2p, theta_0, theta_2):
    inputf1, dt, k2p = inputf1_dt_k2p
    inputf1_dt = (inputf1, dt)
    theta_1 = theta_0 * (k2p - theta_2)
    theta = np.array([theta_0, theta_1, theta_2])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    tac = srtm_fun(inputf1_dt, r1, k2, bp)
    return tac


def srtm_fun_k2p_w(inputf1_dt_k2p_w, theta_0, theta_2):
    inputf1, dt1, dt2, k2p, w = inputf1_dt_k2p_w
    dt = np.array([dt1, dt2])
    k2p = k2p[0]
    tac = srtm_fun_k2p((inputf1, dt, k2p), theta_0, theta_2)
    if w is None:
        w = 1
    tac_w = tac * w
    return tac_w


def srtm_k2p(tac, dt, inputf1, w, k2p):

    inputf1 = kt.int2dt(inputf1, dt)
    k2p = np.full_like(dt[0,], k2p)
    
    if w is None:
        w = 1
    if np.isscalar(w):
        w = np.full_like(dt[0,], w)   

    inputf1_dt_k2p = inputf1, dt, k2p
    inputf1_dt_k2p_w = inputf1, dt[0,], dt[1,], k2p, w # must be k*M matrix
    print(inputf1_dt_k2p_w)
    p, _ = curve_fit(srtm_fun_k2p_w, inputf1_dt_k2p_w, tac * w, p0=(1, 0.5), bounds=(0, [3, 10]))

    theta_0 = p[0]
    theta_2 = p[1]
    theta_1 = theta_0 * (k2p - theta_2)
    theta = np.array([theta_0, theta_1, theta_2])
    r1, k2, bp = kp.srtm_theta2kp(theta)
    tacf = srtm_fun_k2p(inputf1_dt_k2p, theta_0, theta_2)
    kps = {'r1': r1, 'k2': k2, 'bp': bp, 'tacf': tacf}
    return kps


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #  model for ref input
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
def exp_1_fun_t(t, a0, a1, b1):
    cr = a0 + a1 * np.exp(-b1 * t)
    return cr


def exp_1_fun(ts_te_w, a0, a1, b1):
    ts, te, w = ts_te_w
    if w is None:
        w = 1
    cr_dt_fun = a0 * (te - ts) - (a1 * (np.exp(-b1 * te) - np.exp(-b1 * ts))) / b1
    return cr_dt_fun * w


def exp_1(tac, dt, idx, w, fig):
    tac[tac < 0] = 0.0
    if w is None:
        w = np.ones_like(tac)
    ts_te_w = (dt[0, idx], dt[1, idx], w[idx])
    p0 = (3000, 1, 1)
    p, _ = curve_fit(exp_1_fun, ts_te_w, tac[idx] * w[idx], p0)
    a0, a1, b1 = p
    t1 = np.arange(np.amax(dt))
    tac1f = exp_1_fun_t(t1, a0, a1, b1)
    if fig:
        print(p)
        mft = kt.dt2mft(dt)
        plt.plot(t1, tac1f, 'b', mft, tac, 'go')
        plt.show()
    return tac1f, p


def exp_2_fun_t(t, a0, a1, a2, b1, b2):
    cr = a0 + a1 * np.exp(-b1 * t) + a2 * np.exp(-b2 * t)
    return cr


def exp_2_fun(ts_te_w, a0, a1, a2, b1, b2):
    ts, te, w = ts_te_w
    if w is None:
        w = 1
    cr_dt_fun = a0 * (te - ts) - (a1 * (np.exp(-b1 * te) - np.exp(-b1 * ts))) / b1 - (a2 * (np.exp(-b2 * te)
                                                                                            - np.exp(-b2 * ts))) / b2
    return cr_dt_fun * w


def exp_2(tac, dt, idx, w, fig):
    tac[tac < 0] = 0.0
    if w is None:
        w = np.ones_like(tac)
    ts_te_w = (dt[0, idx], dt[1, idx], w[idx])
    p0 = (3000, 1, 1, 1, 1)
    p, _ = curve_fit(exp_2_fun, ts_te_w, tac[idx] * w[idx], p0)
    a0, a1, a2, b1, b2 = p
    t1 = np.arange(np.amax(dt))
    tac1f = exp_2_fun_t(t1, a0, a1, a2, b1, b2)
    if fig:
        print(p)
        mft = kt.dt2mft(dt)
        plt.plot(t1, tac1f, 'b', mft, tac, 'go')
        plt.show()
    return tac1f, p


def exp_am(tac, dt, idx, fig):
    tac[tac < 0] = 0.0
    mft = kt.dt2mft(dt)
    p0 = (0.1, 1, 0.1)
    # p, _ = curve_fit(exp_1_fun_t, mft[idx], tac[idx], p0=p0, bounds=(0.00000001, 2500))
    p, _ = curve_fit(exp_1_fun_t, mft[idx], tac[idx], p0=p0)
    a0, a1, b1 = p
    t1 = np.arange(np.amax(dt))
    tac1f = exp_1_fun_t(t1, a0, a1, b1)
    if fig:
        print(p)
        plt.plot(t1, tac1f, 'b', mft, tac, 'go')
        plt.show()
    return tac1f, p


def feng_srtm_fun(ts_te_w, a0, a1, a2, a3, b0, b1, b2, b3):
    ts, te, w = ts_te_w
    if w is None:
        w = 1
    cr_dt_fun = (a0 * a3 * (b0 ** 2 * te / (
            b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(
        b0 * te)) - b0 * b3 * te / (
                                    b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(
                                b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(
                                b0 * te)) + 2 * b0 / (
                                    b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(
                                b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(
                                b0 * te)) - b3 / (
                                    b0 ** 4 * np.exp(b0 * te) - 2 * b0 ** 3 * b3 * np.exp(
                                b0 * te) + b0 ** 2 * b3 ** 2 * np.exp(
                                b0 * te))) - a0 * a3 * (b0 ** 2 * ts / (
            b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(
        b0 * ts)) - b0 * b3 * ts / (
                                                                b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(
                                                            b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(b0 * ts)) + 2 * b0 / (
                                                                b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(
                                                            b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(b0 * ts)) - b3 / (
                                                                b0 ** 4 * np.exp(b0 * ts) - 2 * b0 ** 3 * b3 * np.exp(
                                                            b0 * ts) + b0 ** 2 * b3 ** 2 * np.exp(
                                                            b0 * ts))) + a0 * a3 * np.exp(-b3 * ts) / (
                         b3 * (b0 ** 2 - 2 * b0 * b3 + b3 ** 2)) - a0 * a3 * np.exp(-b3 * te) / (
                         b3 * (b0 ** 2 - 2 * b0 * b3 + b3 ** 2)) - a1 * a3 * np.exp(-b1 * ts) / (
                         b1 ** 2 - b1 * b3) + a1 * a3 * np.exp(-b1 * te) / (b1 ** 2 - b1 * b3) + a1 * a3 * np.exp(
        -b0 * ts) / (
                         b0 ** 2 - b0 * b3) - a1 * a3 * np.exp(-b0 * te) / (b0 ** 2 - b0 * b3) + a1 * a3 * np.exp(
        -b3 * ts) / (
                         b3 * (b1 - b3)) - a1 * a3 * np.exp(-b3 * te) / (b3 * (b1 - b3)) - a1 * a3 * np.exp(
        -b3 * ts) / (
                         b3 * (b0 - b3)) + a1 * a3 * np.exp(-b3 * te) / (b3 * (b0 - b3)) - a2 * a3 * np.exp(
        -b2 * ts) / (
                         b2 ** 2 - b2 * b3) + a2 * a3 * np.exp(-b2 * te) / (b2 ** 2 - b2 * b3) + a2 * a3 * np.exp(
        -b0 * ts) / (
                         b0 ** 2 - b0 * b3) - a2 * a3 * np.exp(-b0 * te) / (b0 ** 2 - b0 * b3) + a2 * a3 * np.exp(
        -b3 * ts) / (
                         b3 * (b2 - b3)) - a2 * a3 * np.exp(-b3 * te) / (b3 * (b2 - b3)) - a2 * a3 * np.exp(
        -b3 * ts) / (
                         b3 * (b0 - b3)) + a2 * a3 * np.exp(-b3 * te) / (b3 * (b0 - b3))) / (te - ts)

    return cr_dt_fun * w


def feng_fun_t(t, a0, a1, a2, b0, b1, b2):
    cp = a0 * t * np.exp(-b0 * t) + a1 * np.exp(-b1 * t) - a1 * np.exp(-b0 * t) + a2 * np.exp(-b2 * t) - a2 * np.exp(
        -b0 * t)
    return cp


def feng_srtm_fun_t(t, a0, a1, a2, a3, b0, b1, b2, b3):
    cr1 = a0 * a3 * (-b0 * t * np.exp(b3 * t) / (
                b0 ** 2 * np.exp(b0 * t) - 2 * b0 * b3 * np.exp(b0 * t) + b3 ** 2 * np.exp(b0 * t))
                     + b3 * t * np.exp(b3 * t) / (
                                 b0 ** 2 * np.exp(b0 * t) - 2 * b0 * b3 * np.exp(b0 * t) + b3 ** 2 * np.exp(b0 * t))
                     - np.exp(b3 * t) / (b0 ** 2 * np.exp(b0 * t) - 2 * b0 * b3 * np.exp(b0 * t) + b3 ** 2 * np.exp(
                b0 * t))) * np.exp(-b3 * t)

    cr2 = a0 * a3 * np.exp(-b3 * t) / (b0 ** 2 - 2 * b0 * b3 + b3 ** 2)
    cr3 = - a1 * a3 / (b1 * np.exp(b1 * t) - b3 * np.exp(b1 * t))
    cr4 = a1 * a3 / (b0 * np.exp(b0 * t) - b3 * np.exp(b0 * t))
    cr5 = a1 * a3 * np.exp(-b3 * t) / (b1 - b3)
    cr6 = - a1 * a3 * np.exp(-b3 * t) / (b0 - b3)
    cr7 = - a2 * a3 / (b2 * np.exp(b2 * t) - b3 * np.exp(b2 * t))
    cr8 = a2 * a3 / (b0 * np.exp(b0 * t) - b3 * np.exp(b0 * t))
    cr9 = a2 * a3 * np.exp(-b3 * t) / (b2 - b3)
    cr10 = - a2 * a3 * np.exp(-b3 * t) / (b0 - b3)
    cr = np.stack((cr1, cr2, cr3, cr4, cr5, cr6, cr7, cr8, cr9, cr10))
    cr[np.isinf(cr)] = 0
    cr[np.isnan(cr)] = 0
    cr = np.sum(cr, axis=0)
    return cr


def feng_srtm(tac, dt, w, fig):
    tac[tac < 0] = 0.0
    if w is None:
        w = 1
    if np.isscalar(w):
        w = np.full_like(dt[0,], w)
    ts_te_w = (dt[0,], dt[1,], w)
    # p0 = [3.90671734e+00, 4.34910151e+02, 9.22189828e+01, 1.35949657e-02, 4.56109635e-02, 4.53841116e-02, 4.54180443e-02, 7.71163349e-04]
    p0 = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
    # p0 = [6.85579165e-05, -2.08643110e+01, -1.81889002e+02, 7.16906660e+00, 4.21217390e-04, 7.23514957e-02, 7.84986975e-02, 8.27340347e-02]
    p, _ = curve_fit(feng_srtm_fun, ts_te_w, tac * w, p0=p0)
    a0, a1, a2, a3, b0, b1, b2, b3 = p
    t1 = np.arange(np.amax(dt))
    tac1f = feng_srtm_fun_t(t1, a0, a1, a2, a3, b0, b1, b2, b3)
    # print(tac1f)
    if fig:
        print(p)
        cp1f = feng_fun_t(t1, a0, a1, a2, b0, b1, b2)
        mft = kt.dt2mft(dt)
        plt.plot(t1, cp1f, 'r', t1, tac1f, 'b', mft, tac, 'go')
        plt.show()
    return tac1f, p


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # select model args from user_inputs
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_model_inputs(user_inputs, model_name):
    sig = inspect.signature(globals()[model_name])
    model_inputs = dict()
    for p in sig.parameters.values():
        n = p.name
        d = p.default
        if d == inspect.Parameter.empty:
            d = None
        if user_inputs.get(n, 0) is not 0:
            model_inputs.update({n: user_inputs.get(n)})
    return model_inputs
