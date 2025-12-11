#  Three delta robot algos





$\exists (A,B,C,L_1) \in \alpha$ 使得：

1. $\angle AL_1B = \angle BL_1C = \angle CL_1A = \frac{2\pi}{3}$
2. $AL_1 = BL_1 = CL_1 = l_0$
3. $\exists O \notin \alpha$ 使得 $OL_1 \bot \alpha$ 且 $OL_1 = l_1$
4. $\exists L_2 \notin \alpha, L_2 \in \beta, \beta \neq \alpha$ 使得 $OL_2 = l_2$ 且 $OL_2 \bot \beta$
5. $\langle \overrightarrow{L_1O} ,\overrightarrow{OL_2}\rangle > \frac{\pi}{2}$
6. $A,B,C$顺时针排列

$\exists(A',B'，C') \in \beta$使得

$A',B',C'$到$OL_1$距离 $d_{zA'B'C'} = l_0$

$\exists n_{\overrightarrow{L_2A'}}, n_{\overrightarrow{L_2B'}}, n_{\overrightarrow{L_2C'}}, L_2 \in \beta$ 
$\langle n_{\overrightarrow{L_2A'}}, n_{\overrightarrow{L_2B'}}\rangle =\langle n_{\overrightarrow{L_2B'}}, n_{\overrightarrow{L_2C'}}\rangle =\langle n_{\overrightarrow{L_2C'}}, n_{\overrightarrow{L_2A'}}\rangle = \frac{2\pi}{3}$

且满足：

1. $\forall \delta =\langle \overrightarrow{OL_1}, \overrightarrow{OL_2}\rangle \in [-\pi, -\frac{\pi}{2}]$
2. $AA' \bot L_1A, BB' \bot L_1B, CC' \bot L_1C$
3. $AA' \cap n_{\langle L_2, A'\rangle} = A'$ 且 $BB' \cap n_{\langle L_2, B'\rangle} = B'$ 且 $CC' \cap n_{\langle L_2, C'\rangle} = C'$

定义：

$\phi_a =\langle \overrightarrow{OL_1}, \overrightarrow{AA'}\rangle, \phi_b = \langle \overrightarrow{OL_1}, \overrightarrow{BB'}\rangle, \phi_c = \langle \overrightarrow{OL_1}, \overrightarrow{CC'}\rangle$

$\theta_a = rangle \langle \overrightarrow{A'A}, \beta\rangle, \theta_b = rangle \langle \overrightarrow{B'B}, \beta \rangle, \theta_c = rangle \langle \overrightarrow{C'C}, \beta\rangle$

已知 $l_0, l_1, l_2$ 为常量，当 $\delta= -\pi$ 时：

- $L_2A' = L_2B' = L_2C' = l_0$
- $L_2A' \parallel L_1A, L_2B' \parallel L_1B, L_2C' \parallel L_1C$
- $OL_2 \sub L_1L_2 \bot \beta ,OL_2 \sub L_1L_2 \bot\alpha$
- $\alpha \parallel\beta$

且 $L_2A', L_2B', L_2C'$ 随 $\mu$ 改变。

以 $\overrightarrow{L_1O}$ 为 $z$ 轴正方向，以 $\overrightarrow{L_1A}$ 平行，过$O$点的法向量为 $y$ 轴负方向，以过$O$点的平面 平行于$\alpha$ 平面为 $XoY$ 平面建立坐标系。

$\langle \alpha, \beta\rangle = \delta + \pi$ 



$\beta$ 平面以$L_2A' \parallel L_1A, L_2B' \parallel L_1B, L_2C' \parallel L_1C$时作为z轴0旋转，顺时针为正方向，可绕Z轴旋转范围$[-\frac{\pi}{6},\frac{\pi}{6}]$







现有

$\gamma_1 = \overrightarrow {OA} = [0,-l_0,-l_1]^T,\gamma_2=\overrightarrow{OB}=[-l_0\cos{\frac{2\pi}{3}},l_0 \sin{\frac{2\pi}{3}},-l_1]^T,\gamma_3=\overrightarrow{OC}=[l_0\cos{\frac{2\pi}{3}},l_0 \sin{\frac{2\pi}{3}},-l_1]^T$

$\Mu = \begin{bmatrix} \gamma_1 \quad \gamma_2 \quad \gamma_3  \end{bmatrix}$

$\mu = \begin{bmatrix} r\quad p\quad y\end{bmatrix} ^T$，其中$r,p,y \in [-\frac{\pi}{6},\frac{\pi}{6}]$为三个转轴旋转角度，指$OL_2$此杆 ，$r$绕$x$轴旋转，$p$绕$y$轴旋转，$y$绕$z$轴旋转

$\eta_1 =  \overrightarrow {OA'} = [0,-l_0,l_2]^T,\eta_2= \overrightarrow {OB'}=[-l_0\cos{\frac{2\pi}{3}},l_0 \sin{\frac{2\pi}{3}},l_2]^T,\eta_3=\overrightarrow{OC'}=[l_0\cos{\frac{2\pi}{3}},l_0 \sin{\frac{2\pi}{3}},l_2]^T$

$\Nu = \begin{bmatrix} \eta_1\quad\eta_2\quad\eta_3\end{bmatrix}$

当$\mu \to N$（影响$\Nu$）时，求其影响后的$N'$以及根据$N'$求得 $\theta_a, \theta_b, \theta_c$

