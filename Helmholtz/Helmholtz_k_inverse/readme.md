# 利用神经网络反演2D-Helmholtz的波数

- [利用神经网络反演2D-Helmholtz的波数](#利用神经网络反演2d-helmholtz的波数)
  - [问题描述](#问题描述)
  - [环境配置](#环境配置)
  - [优化目标](#优化目标)
  - [数据设计](#数据设计)
  - [研究目标](#研究目标)
    - [改变边界控制点数](#改变边界控制点数)
    - [改变PDE控制点数](#改变pde控制点数)

## 问题描述

As an example, let us consider the Helmholtz equation in two space dimensions:

$$
\Delta{u(x,y)}+k^2u(x,y)=q(x,y),(x,y)\in\Omega:=(-1,1)\\\\
$$

Assuming that the source term is in the form like:

$$
q(x,y)=0
$$

We can easily fabricate an exact solution to this problem:
$$
u(x,y)=0.5*sin(k x)sin(k y)
$$

Sometimes we need to infer the distribution of the medium through the sound field.Now let's consider the following problem: given the source term and boundary conditions of the Helmholtz equation, the wave number in the equation is unknown. Our objective is to infer the distribution of the wave number based on the measured or simulated sound field. However, the measured data is corrupted by measurement noise, and the simulated data is subject to discretization errors. Consequently, the challenge lies in how to perform wave number inversion using these corrupted data, which constitutes a challenging and academically significant inverse problem.


Now let's assume we have a dataset corresponding to a wave number of $k=\sqrt{w/c}=1$. We use 
$$u(x,y)=0.5*sin( x)sin( y)$$

to create our training dataset. The dataset is divided into two parts: one part consists of sampled points representing the boundary conditions, while the other part consists of sampled points from the entire domain. The boundary sampling points impose constraints on the physical model, ensuring that the boundary conditions are satisfied. However, it is also important for the boundary sampling points to satisfy the physical equations. Therefore, the overall domain sampling points should include the boundary sampling points. 

We aim to learn the wave number $k$ through a neural network.There are two ways to achieve this objective.
+ Data-driven method:
  Following a similar approach as in image classification, we adopt the strategy of associating sound field data with corresponding densities obtained through actual testing or simulation. Each set of sound field data is used as input to the network, while the corresponding density serves as the label for training the data-label pairs.This approach has a clear drawback, which is the lack of physical interpretability. Unlike conventional image classification problems, data in physical problems possesses strong constraints and implicit physical models. The purely data-driven approach fails to embed the Helmholtz equation into the neural network to guide the learning process. In addition, due to the continuous variation of wave number with density at a single frequency, a significant number of tests and simulations are required to establish an adequate amount of data-label pairs. This poses disadvantages in terms of cost and time. Therefore, there is a pressing need for a method that can embed the physical model into the neural network while reducing the amount of training data required.
+ Physical Informed Neural Network:
  Physics-Informed Neural Networks (PINNs) are specifically designed to incorporate the physical model into the neural network to guide the learning process.Through this approach, we incorporate the boundary conditions and partial differential equation (PDE) residuals into the loss function for training. Additionally, instead of treating the predicted physical parameters as labels, we consider them as network parameters. Thus, during the optimization process of updating the network parameters, we simultaneously update the physical parameters, effectively treating them as an integral part of the network. This method can significantly reduce the required training data volume.
## 环境配置
    python ==3.9
    pytorch == 2.0.0+cu117
    gpu == NVIDIA GeForce RTX 3050 Ti Laptop GPU(RAM=4GB)

## 优化目标

$f(x) = W^{[n]}(...\sigma(W^{[n-1]}x + b^{[n-1]}))+b^{[n]}$

For a typical initial and boundary value problem,  loss
functions would take the form
$$
L=\lambda_1L_{pde}+\lambda_2L_{data}\\\\
L_{pde}(W,k) =\frac{1}{N_{pde}}\sum_{i=1}^{N_{pde}}[Helmholtz]^2\\\\
L_{data}(W) =\frac{1}{N_{data}}\sum_{i=1}^{N_{data}}[f(x_{i})-u_{i}]^2\\\\
$$


## 数据设计
According to the problem described before , we need to generate the data for PINN. We assume the area is $x\in(-1,1) , y\in(-1,1)$ . It is obviously that the more data points we use, the better the training effect. But huge amount of data will significantly increase the computational cost, thus an appropriate amount of data is necessary.

First , we seperate the area to the shape of 256*256 as test data.

``````python
x_1 = np.linspace(-1,1,256)  # 256 points between -1 and 1 [256x1]
x_2 = np.linspace(1,-1,256)  # 256 points between 1 and -1 [256x1]
``````
Second , we conduct Latin sampling in the interval as PDE point , 'N_f' is the number of sampled data .
``````python
X_f = lb + (ub-lb)*lhs(2,N_f) 
``````
Thirdly , we sample boundary point randomly with the number 'N_u'
``````python
idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False) 
X_u_train = all_X_u_train[idx[0:N_u], :] #choose indices from  set 'idx' (x,t)
u_train = all_u_train[idx[0:N_u],:]      #choose corresponding u
``````

## 研究目标
+ Solve the 2D Helmholtz equation with source correctly ?
+ Time used ?
+ The influence of bc point number and pde point number ?

### 改变边界控制点数
    N_u = 100
    N_f = 10000
    Training time: 21.98
    Test Error: 0.13519
![](picture/20230818021955.png)

    N_u = 200
    N_f = 10000
    Training time: 21.98
    Test Error: 0.10051
![](picture/20230818022205.png)

    N_u = 300
    N_f = 10000
    Training time: 22.74
    Test Error: 0.10007
![](picture/20230818022325.png)

    N_u = 400 
    N_f = 10000
    Training time: 22.23
    Test Error: 0.08802
![](picture/20230818022449.png)


### 改变PDE控制点数
    N_u = 400 
    N_f = 20000
    Training time: 39.04
    Test Error: 0.10225
![](picture/20230818022808.png)

    N_u = 400 
    N_f = 30000
    Training time: 50.46
    Test Error: 0.07301
![](picture/20230818023030.png)

    N_u = 400 
    N_f = 40000
    Training time: 66.36
    Test Error: 0.10518
![](picture/20230818023800.png)

    N_u = 400 
    N_f = 50000
    Training time: 81.16
    Test Error: 0.08737
![](picture/20230818023515.png)


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
