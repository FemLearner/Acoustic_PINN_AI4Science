import torch
import torch.nn as nn     
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        self.layers = layers
        super(DNN, self).__init__()
        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')
    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.iter = 0
        
    
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            
            # weights from a normal distribution with Recommended gain value?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,X):
        a = X.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        out = self.linears[-1](a)
        return out           
    """原版网络"""
    #     # parameters
    #     self.depth = len(layers) - 1

    #     # set up layer order dict
    #     self.activation = torch.nn.Tanh

    #     layer_list = list()
    #     for i in range(self.depth - 1):
    #         layer_list.append(
    #             ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
    #         )
    #         layer_list.append(('activation_%d' % i, self.activation()))

    #     layer_list.append(
    #         ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
    #     )
    #     layerDict = OrderedDict(layer_list)

    #     # deploy layers
    #     self.layers = torch.nn.Sequential(layerDict)

    # def forward(self, x):
    #     out = self.layers(x)
    #     return out
    """新版网络"""
       

# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u,mesh_x,mesh_y, layers, lb, ub,iter_num):
        # mesh
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.x_pde_store = X

        # settings
        self.k = torch.tensor([.1], requires_grad=True).to(device)  # 初始化一个波数（自己选择）
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(device)
        # torch.nn.Parameter()将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数
        # 并将这个参数绑定到module里面，成为module中可训练的参数
        self.k = torch.nn.Parameter(self.k)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        # deep neural networks
        # register_parameter()将一个参数添加到模块中.
        # 通过使用给定的名字可以访问该参数.
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('k', self.k)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=iter_num,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0
        #记录损失
        self.loss_pde = []
        self.loss_data = []
        self.loss_total = []
        self.iter_list = []
        self.k_list = []
    def net_u(self, x, y):
        u = self.dnn(torch.cat([x, y], dim=1))
        return u

    def net_f(self, x, y):
        """ The pytorch autograd version of calculating residual """
        lambda_1 = self.lambda_1        
        lambda_2 = torch.exp(self.lambda_2)
        k = self.k
        #k =torch.exp(self.k)
        u = self.net_u(x, y)

        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_xx + u_yy + k *k * u
        #f = u_y+lambda_1*u*u_x-lambda_2*u_xx
        self.u_x_store = u_x.detach().cpu().numpy()
        self.u_xx_store = u_xx.detach().cpu().numpy()
        self.u_y_store = u_y.detach().cpu().numpy()
        self.u_yy_store = u_yy.detach().cpu().numpy()
        return f

    def loss_func(self):
        u_pred = self.net_u(self.x, self.y)
        f_pred = self.net_f(self.x, self.y)
        loss_pde = torch.mean(f_pred ** 2)
        loss_data = torch.mean((self.u - u_pred) ** 2) 
        loss =loss_data+loss_pde
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if (self.iter % 100 == 0):
            print(
                'Iter : %i, Total_Loss: %e, Pde_Loss: %e, Data_Loss: %e, k: %.5f, lambda1: %.5f, lambda2: %.5f' %
                (   self.iter,
                    loss.item(),
                    loss_pde.item(),
                    loss_data.item(),
                    self.k.item(),
                    #torch.exp(self.k.detach()).item(),
                    self.lambda_1.item(), 
                    torch.exp(self.lambda_2.detach()).item()
                )
            )              
            self.loss_pde.append(loss_pde.detach().cpu().item())
            self.loss_data.append(loss_data.detach().cpu().item())
            self.loss_total.append(loss.detach().cpu().item())
            self.iter_list.append(self.iter)
            self.k_list.append(self.k.detach().cpu().item())
            if(self.if_draw_gra==True):
                self.save_gradient_fig(self.mesh_x,self.mesh_y,self.iter)
        return loss

    def train(self, nIter,if_draw_gra):
        self.dnn.train()
        self.if_draw_gra =if_draw_gra
        """ADAM优化器"""
        # for epoch in range(nIter):
        #     u_pred = self.net_u(self.x, self.y)
        #     f_pred = self.net_f(self.x, self.y)
        #     loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

        #     # Backward and optimize
        #     self.optimizer_Adam.zero_grad()
        #     loss.backward()
        #     self.optimizer_Adam.step()

        #     if epoch % 100 == 0:
        #         print(
        #             'It: %d, Loss: %.3e, k: %.3f' %
        #             (
        #                 epoch,
        #                 loss.item(),
        #                 self.k.item()
        #             )
        #         )

        self.loss_pde = []
        self.loss_data = []
        self.loss_total = []
        self.iter_list = []
        self.k_list = []
        # Backward and optimize牛顿优化器
        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, y)
        f = self.net_f(x, y)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

    def evaluate(self,X_test,u_test,k):
        u_pred, f_pred = self.predict(X_test)

        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)

        k_value = self.dnn.k.detach().cpu().numpy()

        error_k = np.abs(k_value - k) / k * 100

        print('Error u: %e' % (error_u))
        print('Error k: %.5f%%' % (error_k))

    def save_gradient_fig(self,mesh_x,mesh_y,iter):
            ###########u_x#################
        value_griddata = griddata(self.x_pde_store, self.u_x_store.flatten(), (mesh_x, mesh_y), method='cubic')
        fig = plt.figure(figsize=(50,12))
        ax = fig.add_subplot(141)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        value_griddata = griddata(self.x_pde_store, self.u_xx_store.flatten(), (mesh_x, mesh_y), method='cubic')
        ax = fig.add_subplot(142)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        ax.set_title('$u_{xx}pred$' , fontsize=15)  # font size doubled
        ax.tick_params(labelsize=15)
        ###########u_y#################
        value_griddata = griddata(self.x_pde_store, self.u_y_store.flatten(), (mesh_x, mesh_y), method='cubic')
        ax = fig.add_subplot(143)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        value_griddata = griddata(self.x_pde_store, self.u_yy_store.flatten(), (mesh_x, mesh_y), method='cubic')
        ax = fig.add_subplot(144)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        ax.set_title('$u_{yy}pred$' , fontsize=15)  # font size doubled
        ax.tick_params(labelsize=15)
        plt.title('iter-%iter'%(iter),loc='center')
        plt.savefig('./Gradient_train_picture/Gradient-iter-%i.png'%(iter), dpi=100, bbox_inches='tight')
       
        list_filename = "./Gradient_train_picture/list.txt"
        f = open(list_filename,'a')
        f.write('Gradient-iter-%i.png'%(iter))
        f.write('\n')
        f.close


        # the physics-guided neural network
class PhysicsInformedNN2():
    def __init__(self, X, u,mesh_x,mesh_y, layers, lb, ub,iter_num):
        # mesh
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.x_pde_store = X

        # settings
        self.k = torch.tensor([.1], requires_grad=True).to(device)  # 初始化一个波数（自己选择）

        # torch.nn.Parameter()将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数
        # 并将这个参数绑定到module里面，成为module中可训练的参数
        self.k = torch.nn.Parameter(self.k)
        # deep neural networks
        # register_parameter()将一个参数添加到模块中.
        # 通过使用给定的名字可以访问该参数.
        self.dnn = DNN(layers).to(device)
        # self.dnn.register_parameter('k', self.k)
        #拟合k的网络
        self.dnnk = DNN(layers).to(device)
        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=iter_num,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0
        #记录损失
        self.loss_pde = []
        self.loss_data = []
        self.loss_total = []
        self.iter_list = []
        self.k_list = []
    def net_u(self, x, y):
        u = self.dnn(torch.cat([x, y], dim=1))
        return u
    def net_k(self, x, y):
        kpred = self.dnnk(torch.cat([x, y], dim=1))
        return kpred
    def net_f(self, x, y):
        """ The pytorch autograd version of calculating residual """

        k = self.net_k(x, y)
        #k =torch.exp(self.k)
        u = self.net_u(x, y)

        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_xx + u_yy + k *k * u
        #f = u_y+lambda_1*u*u_x-lambda_2*u_xx
        self.u_x_store = u_x.detach().cpu().numpy()
        self.u_xx_store = u_xx.detach().cpu().numpy()
        self.u_y_store = u_y.detach().cpu().numpy()
        self.u_yy_store = u_yy.detach().cpu().numpy()
        return f

    def loss_func(self):
        u_pred = self.net_u(self.x, self.y)
        k = self.net_k(self.x, self.y)
        f_pred = self.net_f(self.x, self.y)
        loss_pde = torch.mean(f_pred ** 2)
        loss_data = torch.mean((self.u - u_pred) ** 2) 
        loss =0.01*loss_data+loss_pde
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if (self.iter % 100 == 0):
            print(
                'Iter : %i, Total_Loss: %e, Pde_Loss: %e, Data_Loss: %e, k: %.5f' %
                (   self.iter,
                    loss.item(),
                    loss_pde.item(),
                    loss_data.item(),
                    k.detach().cpu().numpy()[1],
                    #torch.exp(self.k.detach()).item(),

                )
            )              
            self.loss_pde.append(loss_pde.detach().cpu().item())
            self.loss_data.append(loss_data.detach().cpu().item())
            self.loss_total.append(loss.detach().cpu().item())
            self.iter_list.append(self.iter)
            self.k_list.append(self.k.detach().cpu().item())
            if(self.if_draw_gra==True):
                self.save_gradient_fig(self.mesh_x,self.mesh_y,self.iter)
        return loss

    def train(self, nIter,if_draw_gra):
        self.dnn.train()
        self.if_draw_gra =if_draw_gra
        """ADAM优化器"""
        # for epoch in range(nIter):
        #     u_pred = self.net_u(self.x, self.y)
        #     f_pred = self.net_f(self.x, self.y)
        #     loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

        #     # Backward and optimize
        #     self.optimizer_Adam.zero_grad()
        #     loss.backward()
        #     self.optimizer_Adam.step()

        #     if epoch % 100 == 0:
        #         print(
        #             'It: %d, Loss: %.3e, k: %.3f' %
        #             (
        #                 epoch,
        #                 loss.item(),
        #                 self.k.item()
        #             )
        #         )

        self.loss_pde = []
        self.loss_data = []
        self.loss_total = []
        self.iter_list = []
        self.k_list = []
        # Backward and optimize牛顿优化器
        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, y)
        f = self.net_f(x, y)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

    def evaluate(self,X_test,u_test,k):
        u_pred, f_pred = self.predict(X_test)

        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)

        k_value = self.dnn.k.detach().cpu().numpy()

        error_k = np.abs(k_value - k) / k * 100

        print('Error u: %e' % (error_u))
        print('Error k: %.5f%%' % (error_k))

    def save_gradient_fig(self,mesh_x,mesh_y,iter):
            ###########u_x#################
        value_griddata = griddata(self.x_pde_store, self.u_x_store.flatten(), (mesh_x, mesh_y), method='cubic')
        fig = plt.figure(figsize=(50,12))
        ax = fig.add_subplot(141)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        value_griddata = griddata(self.x_pde_store, self.u_xx_store.flatten(), (mesh_x, mesh_y), method='cubic')
        ax = fig.add_subplot(142)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        ax.set_title('$u_{xx}pred$' , fontsize=15)  # font size doubled
        ax.tick_params(labelsize=15)
        ###########u_y#################
        value_griddata = griddata(self.x_pde_store, self.u_y_store.flatten(), (mesh_x, mesh_y), method='cubic')
        ax = fig.add_subplot(143)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        value_griddata = griddata(self.x_pde_store, self.u_yy_store.flatten(), (mesh_x, mesh_y), method='cubic')
        ax = fig.add_subplot(144)
        h = ax.imshow(value_griddata, interpolation='nearest', cmap='rainbow',
                        extent=[mesh_x.min(), mesh_x.max(), mesh_y.min(), mesh_y.max()],
                        origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.plot(
            self.x_pde_store[:, 0],
            self.x_pde_store[:, 1],
            'kx', label='Data (%d points)' % (self.u_x_store.flatten().shape[0]),
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
        ax.set_title('$u_{yy}pred$' , fontsize=15)  # font size doubled
        ax.tick_params(labelsize=15)
        plt.title('iter-%iter'%(iter),loc='center')
        plt.savefig('./Gradient_train_picture/Gradient-iter-%i.png'%(iter), dpi=100, bbox_inches='tight')
       
        list_filename = "./Gradient_train_picture/list.txt"
        f = open(list_filename,'a')
        f.write('Gradient-iter-%i.png'%(iter))
        f.write('\n')
        f.close