 一、正向运动学推导（Forward Kinematics）                                                  
                                                                                            
  问题描述                                                                                  
                                                                                            
  已知：姿态角 $\mu = [r, p, y]^T$                                                          
  求：$A', B', C'$ 的位置和杆的姿态角 $\theta_a, \theta_b, \theta_c, \phi_a, \phi_b,        
  \phi_c$                                                                                   
                                                                                            
  推导步骤                                                                                  
                                                                                            
  1. 旋转矩阵构造                                                                           
                                                                                            

  使用ZYX欧拉角（Tait-Bryan angles）：                                                      
                                                                                            
  $$R(\mu) = R_z(y) \cdot R_y(p) \cdot R_x(r)$$                                             
                                                                                            
  展开形式：                                                                                
                                                                                            
  $$R_x(r) = \begin{bmatrix}                                                                
  1 & 0 & 0 \\                                                                                
  0 & \cos r & -\sin r \\                                                                     
  0 & \sin r & \cos r                                                                       
  \end{bmatrix}$$                                                                           
                                                                                            
  $$R_y(p) = \begin{bmatrix}                                                                
  \cos p & 0 & \sin p \\                                                                      
  0 & 1 & 0             \\                                                                    
  -\sin p & 0 & \cos p    \\                                                                  
  \end{bmatrix}$$                                                                           
                                                                                            
  $$R_z(y) = \begin{bmatrix}                                                                
  \cos y & -\sin y & 0       \\                                                               
  \sin y & \cos y & 0          \\                                                             
  0 & 0 & 1                                                                                 
  \end{bmatrix}$$                                                                           
                                                                                            
  组合后：                                                                                  
                                                                                            
  $$R(\mu) = \begin{bmatrix}                                                                
  c_y c_p & c_y s_p s_r - s_y c_r & c_y s_p c_r + s_y s_r   \\                                
  s_y c_p & s_y s_p s_r + c_y c_r & s_y s_p c_r - c_y s_r     \\                              
  -s_p & c_p s_r & c_p c_r                                                                  
  \end{bmatrix}$$                                                                           
                                                                                            
  其中 $c_\theta = \cos\theta, s_\theta = \sin\theta$。                                     
                                                                                            

  2. 静平台基础位置                                                                         
                                                                                            

  $$\Nu = \begin{bmatrix}                                                                   
  0 & -\frac{\sqrt{3}l_0}{2} & \frac{\sqrt{3}  l_0}{2}      \\                                 
  -l_0 & \frac{l_0}{2} & \frac{l_0}{2}                       \\                               
  l_2 & l_2 & l_2                                                                           
  \end{bmatrix}$$                                                                           
                                                                                            
  简化表示：                                                                                
  $$\eta_1 = [0, -l_0, l_2]^T, \quad \eta_2 = [-\frac{\sqrt{3}}{2}l_0, \frac{1}{2}l_0,      
  l_2]^T, \quad \eta_3 = [\frac{\sqrt{3}}{2}l_0, \frac{1}{2}l_0, l_2]^T$$                   
                                                                                            

  3. 旋转后的静平台位置                                                                     
                                                                                            

  $$\Nu' = R(\mu) \cdot \Nu$$                                                               
                                                                                            
  $$\eta_i' = R(\mu) \cdot \eta_i, \quad i = 1,2,3$$                                        
                                                                                            
  4. 垂直约束求解                                                                           
                                                                                            

  对于每根杆 $i$，有两个垂直约束：                                                          
                                                                                            
  约束1（动平台侧）：杆 $\overrightarrow{AA'}$ 垂直于半径 $\overrightarrow{L_1A}$           
  $$(\eta_i' - \gamma_i) \cdot \gamma_i = 0$$                                               
                                                                                            
  约束2（静平台侧）：杆 $\overrightarrow{A'A}$ 垂直于半径 $\overrightarrow{L_2A'}$          
  $$(\gamma_i - \eta_i') \cdot \eta_i' = 0$$                                                
                                                                                            
  然而，由于机构约束，$\eta_i'$ 实际上不在球面 $|\overrightarrow{O\eta_i'}| =               
  \sqrt{l_0^2 + l_2^2}$ 上，而是被杆长约束。                                                
                                                                                            
  5. 求解方法                                                                           
                                                                                            

  考虑到垂直约束，点 $A'$ 必须满足：                                                        
  - 在过 $\gamma_i$ 且垂直于 $\gamma_i$ 的平面上                                            
  - 在过 $O$ 点，法向量为 $R(\mu) \cdot [0,0,1]^T$ 的平面 $\beta$ 上                        
                                                                                            

  平面 $\beta$ 方程：                                                                       
  $$n_\beta = R(\mu) \cdot \begin{bmatrix} 0\quad  0\quad  1 \end{bmatrix} = \begin{bmatrix}          
  R_{13} \quad R_{23} \quad R_{33} \end{bmatrix}$$                                                    
                                                                                            
  $$n_\beta \cdot \vec{OP} = l_2 \cdot |n_\beta|_z \text{ component}$$                      
                                                                                            
  实际上，由于 $L_2$ 的位置为 $\vec{OL_2} = l_2 \cdot n_\beta$：                            
                                                                                            
  $$\vec{OL_2} = l_2 \cdot R(\mu) \cdot \begin{bmatrix} 0  0  1 \end{bmatrix}$$             
                                                                                            
  点 $A'$ 相对于 $L_2$ 的位置：                                                             
  $$\vec{L_2A'} = R_z(\mu_z) \cdot \vec{L_2A}_{init}$$                                      
                                                                                            
  其中 $\mu_z$ 是 $\beta$ 平面绕其法向量的旋转角（额外自由度，范围 $[-\frac{\pi}{6},        
  \frac{\pi}{6}]$）。                                                                       
                                                                                            
  完整表达式：                                                                              
  $$\eta_i' = \vec{OL_2} + R_\beta(\mu_z) \cdot \eta_i^{local}$$                            
                                                                                            
  其中 $R_\beta(\mu_z)$ 是在 $\beta$ 平面内的旋转矩阵。                                     
                                                                                            

  6. 杆长约束                                                                               
                                                                                            

  $$s_i = |\eta_i' - \gamma_i|$$                                                            
                                                                                            
  这是隐式约束，在正向运动学中作为验证条件。                                                
                                                                                            
  7. 角度计算                                                                               
                                                                                            

  $\theta$ 角（杆与 $\beta$ 平面的夹角）：                                                  
  $$\theta_i = \frac{\pi}{2} - \arccos\left(\frac{(\eta_i' - \gamma_i) \cdot                
  n_\beta}{|\eta_i' - \gamma_i|}\right)$$                                                   
                                                                                            
  $\phi$ 角（杆与 $\overrightarrow{OL_1}$ 的夹角）：                                        
  $$\phi_i = \arccos\left(\frac{(\eta_i' - \gamma_i) \cdot [0,0,1]^T}{|\eta_i' -            
  \gamma_i|}\right)$$                                                                       
                                                                                            

---
  二、逆向运动学推导（Inverse Kinematics）                                                  
                                                                                            
  问题描述                                                                                  
                                                                                            
  已知：目标姿态角 $\theta_a, \theta_b, \theta_c$（或目标位置 $\eta_1', \eta_2',            
  \eta_3'$）                                                                                
  求：对应的 $\mu = [r, p, y]^T$                                                            
                                                                                            
  推导步骤                                                                                  
                                                                                            

  1. 约束方程组                                                                             
                                                                                            

  对于每根杆 $i$，建立约束方程：                                                            
                                                                                            
  $$f_i(\mu) = (\eta_i'(\mu) - \gamma_i) \cdot \gamma_i = 0$$                               
                                                                                            
  这是3个非线性方程，3个未知数。                                                            
                                                                                            

  2. Newton-Raphson迭代                                                                     
                                                                                            

  $$\mu^{(k+1)} = \mu^{(k)} - J^{-1}(\mu^{(k)}) \cdot F(\mu^{(k)})$$                        
                                                                                            
  其中：                                                                                    
  - $F(\mu) = [f_1(\mu), f_2(\mu), f_3(\mu)]^T$                                             
  - $J(\mu)$ 是雅可比矩阵                                                                   
                                                                                            
  3. 雅可比矩阵推导                                                                         
                                                                                            

  $$J_{ij} = \frac{\partial f_i}{\partial \mu_j}$$                                          
                                                                                            
  展开 $f_i(\mu)$：                                                                         
                                                                                            
  $$f_i(\mu) = (\eta_i'(\mu) - \gamma_i) \cdot \gamma_i$$                                   
                                                                                            
  $$= (R(\mu) \eta_i - \gamma_i) \cdot \gamma_i$$                                           
                                                                                            
  $$= \eta_i^T R(\mu)^T \gamma_i - \gamma_i^T \gamma_i$$                                    
                                                                                            
  求偏导：                                                                                  
                                                                                            
  $$\frac{\partial f_i}{\partial \mu_j} = \eta_i^T \frac{\partial R(\mu)^T}{\partial        
  \mu_j} \gamma_i$$                                                                         
                                                                                            
  旋转矩阵的偏导：                                                                          
                                                                                            
  $$\frac{\partial R}{\partial r} = R_z(y) \cdot R_y(p) \cdot \frac{\partial                
  R_x(r)}{\partial r}$$                                                                     
                                                                                            
  $$\frac{\partial R_x(r)}{\partial r} = \begin{bmatrix}                                    
  0 & 0 & 0         \\                                                                        
  0 & -\sin r & -\cos r\\                                                                     
  0 & \cos r & -\sin r                                                                      
  \end{bmatrix}$$                                                                           
                                                                                            
  类似地计算 $\frac{\partial R}{\partial p}$ 和 $\frac{\partial R}{\partial y}$。           
                                                                                            

  4. 完整雅可比矩阵                                                                                                                                  
    
  $$J(\mu) = \begin{bmatrix}                                                                
    \eta_1^T \frac{\partial R^T}{\partial r} \gamma_1 & \eta_1^T \frac{\partial               
    R^T}{\partial p} \gamma_1 & \eta_1^T \frac{\partial R^T}{\partial y} \gamma_1  \\           
    \eta_2^T \frac{\partial R^T}{\partial r} \gamma_2 & \eta_2^T \frac{\partial               
    R^T}{\partial p} \gamma_2 & \eta_2^T \frac{\partial R^T}{\partial y} \gamma_2    \\         
    \eta_3^T \frac{\partial R^T}{\partial r} \gamma_3 & \eta_3^T \frac{\partial               
    R^T}{\partial p} \gamma_3 & \eta_3^T \frac{\partial R^T}{\partial y} \gamma_3             
    \end{bmatrix}$$  

