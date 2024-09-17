import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings
import os
import shutil
import imageio



def plot_cloud_picture(X_vec,value_vec,mesh_x,mesh_y,k_value,k_true):
    
    value_griddata = griddata(X_vec, value_vec.flatten(), (mesh_x, mesh_y), method='cubic')

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.plot(
        X_vec[:, 0],
        X_vec[:, 1],
        'kx', label='Data (%d points)' % (value_vec.shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )

    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u(x,y)$,  k=%.7f' % (k_value), fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)

    plt.show()

    plt.savefig('Helmholtz_inverse_%.7f.png'%(k_true), dpi=500, bbox_inches='tight')

def plot_gradient_true(mesh_x,mesh_y,k_true):

    u_x = k_true*np.cos(k_true*mesh_x) * np.sin(k_true*mesh_y) #网格化坐标上的真解
    u_xx = -k_true**2*np.sin(k_true*mesh_x) * np.sin(k_true*mesh_y) #网格化坐标上的真解
    u_y = k_true*np.sin(k_true*mesh_x) * np.cos(k_true*mesh_y) #网格化坐标上的真解
    u_yy = -k_true**2*np.sin(k_true*mesh_x) * np.sin(k_true*mesh_y) #网格化坐标上的真解
    X = np.hstack((mesh_x.flatten()[:,None], mesh_y.flatten()[:,None]))
    ###########u_x#################
    value_griddata = griddata(X, u_x.flatten(), (mesh_x, mesh_y), method='cubic')
    fig = plt.figure(figsize=(30,5))
    ax = fig.add_subplot(141)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_x.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_x$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
###########u_xx#################
    value_griddata = griddata(X, u_xx.flatten(), (mesh_x, mesh_y), method='cubic')
    fig = plt.figure(figsize=(30,5))
    ax = fig.add_subplot(142)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_xx.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_xx$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
###########u_y#################
    value_griddata = griddata(X, u_y.flatten(), (mesh_x, mesh_y), method='cubic')
    fig = plt.figure(figsize=(30,5))
    ax = fig.add_subplot(143)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_y.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_y$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
    ###########u_yy#################
    value_griddata = griddata(X, u_yy.flatten(), (mesh_x, mesh_y), method='cubic')
    fig = plt.figure(figsize=(30,5))
    ax = fig.add_subplot(144)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_yy.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_yy$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)

    plt.show()

    plt.savefig('Gradient_TRUE.png', dpi=500, bbox_inches='tight')
    
def plot_gradient(model,mesh_x,mesh_y,k_true):
    ###########u_x#################
    value_griddata = griddata(model.x_pde_store, model.u_x_store.flatten(), (mesh_x, mesh_y), method='cubic')
    fig = plt.figure(figsize=(40,30))
    ax = fig.add_subplot(341)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                    extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        model.x_pde_store[:, 0],
        model.x_pde_store[:, 1],
        'kx', label='Data (%d points)' % (model.u_x_store.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_xpred$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
    ###########u_xx#################
    value_griddata = griddata(model.x_pde_store, model.u_xx_store.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(342)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                    extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        model.x_pde_store[:, 0],
        model.x_pde_store[:, 1],
        'kx', label='Data (%d points)' % (model.u_x_store.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_xxpred$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
    ###########u_y#################
    value_griddata = griddata(model.x_pde_store, model.u_y_store.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(343)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                    extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        model.x_pde_store[:, 0],
        model.x_pde_store[:, 1],
        'kx', label='Data (%d points)' % (model.u_x_store.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_ypred$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
    ###########u_yy#################
    value_griddata = griddata(model.x_pde_store, model.u_yy_store.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(344)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                    extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        model.x_pde_store[:, 0],
        model.x_pde_store[:, 1],
        'kx', label='Data (%d points)' % (model.u_x_store.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_yypred$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
################################################
    u_x = k_true*np.cos(k_true*model.x_pde_store[:,0]) * np.sin(k_true*model.x_pde_store[:,1]) #网格化坐标上的真解
    u_xx = -k_true**2*np.sin(k_true*model.x_pde_store[:,0]) * np.sin(k_true*model.x_pde_store[:,1]) #网格化坐标上的真解
    u_y = k_true*np.sin(k_true*model.x_pde_store[:,0]) * np.cos(k_true*model.x_pde_store[:,1]) #网格化坐标上的真解
    u_yy = -k_true**2*np.sin(k_true*model.x_pde_store[:,0]) * np.sin(k_true*model.x_pde_store[:,1]) #网格化坐标上的真解
    X = model.x_pde_store
    ###########u_x#################
    value_griddata = griddata(X, u_x.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(345)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_x.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_x$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
###########u_xx#################
    value_griddata = griddata(X, u_xx.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(346)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_xx.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_xx$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
###########u_y#################
    value_griddata = griddata(X, u_y.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(347)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_y.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_y$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
    ###########u_yy#################
    value_griddata = griddata(X, u_yy.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(348)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_yy.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_yy$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
################################################err
    u_x_err = np.abs(model.u_x_store.flatten() - k_true*np.cos(k_true*model.x_pde_store[:,0]) * np.sin(k_true*model.x_pde_store[:,1])) #网格化坐标上的真解
    u_xx_err = np.abs(model.u_xx_store.flatten() + k_true**2*np.sin(k_true*model.x_pde_store[:,0]) * np.sin(k_true*model.x_pde_store[:,1])) #网格化坐标上的真解
    u_y_err = np.abs(model.u_y_store.flatten()-k_true*np.sin(k_true*model.x_pde_store[:,0]) * np.cos(k_true*model.x_pde_store[:,1])) #网格化坐标上的真解
    u_yy_err = np.abs(model.u_yy_store.flatten()+k_true**2*np.sin(k_true*model.x_pde_store[:,0]) * np.sin(k_true*model.x_pde_store[:,1])) #网格化坐标上的真解
    X = model.x_pde_store
    ###########u_x#################
    value_griddata = griddata(X, u_x_err.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(3,4,9)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_x_err.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_xerr$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
###########u_xx#################
    value_griddata = griddata(X, u_xx_err.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(3,4,10)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_xx_err.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_xxerr$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
###########u_y#################
    value_griddata = griddata(X, u_y_err.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(3,4,11)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_y_err.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_yerr$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)
    ###########u_yy#################
    value_griddata = griddata(X, u_yy_err.flatten(), (mesh_x, mesh_y), method='cubic')
    ax = fig.add_subplot(3,4,12)
    h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                  extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.plot(
        X[:, 0],
        X[:, 1],
        'kx', label='Data (%d points)' % (u_yy_err.flatten().shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=.5
    )
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$y$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u_yyerr$' , fontsize=15)  # font size doubled
    ax.tick_params(labelsize=15)

    plt.savefig('Gradient.png', dpi=500, bbox_inches='tight')
    plt.show()

def plot_inverse_solu(X,Y,u_test,u_test_pred,k_inverse,k_true):


    #Ground truth
    fig_1 = plt.figure(1, figsize=(30, 25))
    plt.subplot(2, 2, 1)
    plt.pcolor(X, Y, u_test, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.title('Ground Truth $u(x_1,x_2)$, k_true = %.4f'%(k_true), fontsize=15)

    # Prediction
    plt.subplot(2, 2, 2)
    plt.pcolor(X, Y, u_test_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.title('Predicted $\hat u(x_1,x_2)$, k_inverse = %4f'%(k_inverse), fontsize=15)

    # Error
    plt.subplot(2, 2, 3)
    plt.pcolor(X, Y, np.abs(u_test - u_test_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.title(r'Absolute error $|u(x_1,x_2)- \hat u(x_1,x_2)|$ k_true = %.4f,k_inverse = %4f'%(k_true,k_inverse), fontsize=15)
    plt.tight_layout()

    # Prediction
    plt.subplot(2, 2, 4)
    plt.pcolor(X, Y, u_inv_true, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.title('Inv Ground Truth $\hat u(x_1,x_2)$, k_inverse = %4f'%(k_inverse), fontsize=15)

    plt.savefig('Helmholtz_forward.png', dpi = 500, bbox_inches='tight')

def generate_gradient_ani(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    picture_list = []
    for root,folders,files in os.walk(path):
        for file in files:
            picture_list.append(path+file)

    generated_images=[]
    for png_path in picture_list:
        generated_images.append(imageio.imread(png_path))
    # shutil.rmtree(PNGFILE)  # 可删掉
    imageio.mimsave('Gradient.gif', generated_images, 'GIF', duration=0.1)

