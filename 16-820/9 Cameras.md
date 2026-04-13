# Properties of Camera Matrices
A $3 \times 4$ matrix $M$ can be a camera matrix iff $\text{det}(A) \neq 0$.

# Simplified Camera Models
## Orthographic Projection
Rays from the scene to the image plane are parallel to the optical axis.
![[orthographic-projection.png]]
$$x = X, \ y = Y$$
Therefore:
$$M = \begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}$$
Camera projection is linear.
## Scaled Orthographic (Weak Perspective) Projection
Consider two points at different depth that are far away from the camera:
![[weak-perspective-projection.png]]
$$A = \begin{bmatrix}
A_x\\
A_y\\
Z
\end{bmatrix}, \ B = \begin{bmatrix}
B_x\\
B_y\\
Z + \Delta Z 
\end{bmatrix}$$
Therefore:
$$a_x = \frac{fA_x}{Z} = \alpha A_x$$
$$b_x = \frac{f B_x}{Z + \Delta Z} \approx \frac{f B_x}{Z} = \alpha B_x$$
Therefore:
$$M = \begin{bmatrix}
\alpha & 0 & 0 & 0\\
0 & \alpha & 0 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}$$

Weak perspective projection works when changes in depth $\Delta Z$ is small relative to distance from camera $Z$.
## Paraperspective Projection
Rays from the scene to the image plane are parallel to the line from the camera center to a reference point (e.g. the object centroid), rather than the optical axis.
![[paraperspective-projection.png]]

Paraperspective is more accurate than weak perspective for off-axis objects.
## Affine Projection
A general linear camera model.
$$M = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & b_1\\
a_{21} & a_{22} & a_{23} & b_2\\
0 & 0 & 0 & 1
\end{bmatrix}$$

Affine projections:
- Perspective.
- Weak perspective.
- Paraperspective.

# Camera Calibration
![[camera-calibration.png]]

Overall approach:
1. Compute algebraic solution.
	1. Uncalibrated PnP.
	2. Zhang's method.
2. Use algebraic solution to initialize gradient descent over reprojection error.

**Uncalibrated Perspective n-Point (Uncalibrated PnP):**
Uncalibrated Perspective n-Point solves for camera matrix given known point correspondences.

Given knowns $X_i$ and $x_i$:
$$x_i = \frac{m_{11}X_i + m_{12}Y_i + m_{13}Z_i + m_{14}}{m_{31}X_i + m_{32} Y_i + m_{33} Z_i + m_{34}}$$
$$(m_{31}X_i + m_{32} Y_i + m_{33} Z_i + m_{34}) x_i = m_{11}X_i + m_{12}Y_i + m_{13}Z_i + m_{14}$$
$$\begin{bmatrix}
-X_i & -Y_i & -Z_i & -1 & 0 & 0 & 0 & 0 & X_i x_i & Y_i x_i & Z_i x_i & x_i
\end{bmatrix} \begin{bmatrix} 
m_{11}\\
m_{12}\\
m_{13}\\
m_{14}\\
m_{21}\\
m_{22}\\
m_{23}\\
m_{24}\\
m_{31}\\
m_{32}\\
m_{33}\\
m_{34}\\
\end{bmatrix}$$
Therefore:
$$Am = \begin{bmatrix} 0 \\ 0 \\ \vdots \end{bmatrix}$$
- The DOF of $m$ is 11 (because $M$ and $\alpha M$ are the same camera matrix).
- Therefore, we need 6 pairs of corresponding points to solve for H. 

We can then decompose the estimated $M = K [R \ t]$ with **QR decomposition**.

**Perspective n-Point (PnP):**
Perspective n-Point solves for extrinsic $[R \ t]$ given known intrinsic $K$ and point correspondences.
1. Compute DLT solution.
2. **Project solution onto $SO(3)$ via SVD.**
$$R = UV^T, \text{where} \ \tilde R = U \Sigma V^T$$

**Zhang's method**:
1. Estimate intrinsic $K$ with a planar checkerboard by exploiting homography constraints.
2. Estimate extrinsic $[R \ t]$ with PnP.
3. Refine with gradient descent over reprojection error.
	1. Intrinsic.
	2. Extrinsic.
	3. Lens distortion coefficients.

**Lens distortion:**
Lens distortion is a geometric aberration where points in an image are displaced from where an ideal pinhole camera would project them, caused by physical imperfections in the lens.
- Radial lens distortion.
- Tangent lens distortion.

Lens distortion is applied in **normalized image plane**.
$$\underbrace{(X,Y,Z)}_{\text{3D camera space}} \xrightarrow{\div Z} \underbrace{(x,y)}_{\text{normalized}} \xrightarrow{\text{distort}} \underbrace{(x_d, y_d)}_{\text{distorted normalized}} \xrightarrow{K} \underbrace{(u,v)}_{\text{pixel}}$$

**Radial lens distortion:**
Radial lens distortion is a geometric aberration where straight lines in the real world appear curved in an image, caused by the imperfect shape of a camera lens bending light non-uniformly as you move away from the optical axis. 

![[radial-lens-distortion.png]]

The distortion is called "radial" because its magnitude depends only on the radial distance $r$ from the image center:
$$x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6 + \cdots)$$
$$y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6 + \cdots)$$

**Tangent lens distortion:**
Tangential lens distortion is a geometric aberration where points in an image are shifted asymmetrically, caused by the lens not being perfectly parallel to the sensor plane.