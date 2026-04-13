# Pinhole Optics
## Aperture and Lens
Aperture:
- Larger aperture: Brighter, blurrier image.
- Smaller aperture: Dimmer, sharper image.

Lens: Collect more light by focusing them on the same point, leads to brighter and sharper image.

Aperture + lens: The aperture controls depth of field.

**Depth of field**: How much depth exists within the field of view over which objects appear acceptably sharp.
![[dof1.png]]
![[dof2.png]]
## Pinhole Cameras
A pinhole camera consists of a light-proof box with an **infinitely small** hole on one side. 

Because the hole is infinitely small, exactly one ray from each point passes through it, so each point in the scene maps to exactly one point on the image plane. This means the entire scene is in focus at all depths simultaneously.
- No lens.
- No aperture.
- No depth of field.
## Perspective Projection
![[pinhole-camera.png]]
$$x = -\frac{fX}{Z}, \ y = -\frac{fY}{Z}$$
- $f$: Focal length, in pixels.

The image is inverted.

Instead of the real image plane, sometimes we project the image onto an "**easel**" to avoid the inversion. The distance from **center of projection (COP)** to easel is called **focal length**.
![[easel.png]]
## Field of View (FOV)
For a spherical easel:
![[easel-fov.png]]
$$\theta = \frac{L}{f}$$

For a sensored image:
$$\text{FOV} \approx \frac{\text{total sensor size}}{\text{focal length}}$$

Hitchcock zoom:
- Increase focal length and step back.
- FOV decreases.
- Size of object in image remains the same.
## Properties of Perspective Projection
Properties:
- Closer objects appear larger.
- Closer objects are lower in the image.
- Parallel lines meet.
	- The vanishing points depend only on the direction of the line, which is the same for parallel lines.
	- All 3D lines on ground plane meet at a set of vanishing points called the horizon line. The horizon line is always at the center of the image.

Proof:
The 3D line can be expressed as a starting point and a direction:
$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} A_x \\ A_y \\ A_z \end{bmatrix} + \lambda \begin{bmatrix} D_x \\ D_y \\ D_z \end{bmatrix}$$
Therefore:
$$x = \frac{fX}{Z} = \frac{f(A_x + \lambda D_x)}{A_z + \lambda D_z} \to \frac{fD_x}{D_z} \text{ as } \lambda \to \infty$$
Therefore, the vanishing point is:
$$(x, y) \to \left(\frac{fD_x}{D_z}, \frac{fD_y}{D_z}\right) \text{ as } \lambda \to \infty$$

# Camera Matrices
Projective projection in homogeneous form:
$$\lambda \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} fX \\ fY \\ Z \end{bmatrix} = \begin{bmatrix} f & 0 & 0 \\ 0 & f & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

World to camera transformation:
![[w2c.png]]
$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} & r_{32} & r_{33} \end{bmatrix} \left(\begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} - \begin{bmatrix} C_x \\ C_y \\ C_z \end{bmatrix}\right)$$
Can be interpreted as:
1. Computing camera to point offset.
2. Projecting the offset onto the basis vectors of the camera coordinate system.

Formally:
$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = [R \ t] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
Where: $R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} & r_{32} & r_{33} \end{bmatrix}$ and $t = \begin{bmatrix} t_x \\ t_y \\ t_z \end{bmatrix} = -RC$.

Full camera projection:
$$\lambda \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} f & 0 & 0 \\ 0 & f & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

Fancier intrinsics:
$$K = \begin{bmatrix} f_x & s_\theta & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
- Shifted origin: $c_x \neq 0$, $c_y \neq 0$.
- Non-square pixels: $f_x \neq f_y$.
- Skewed image axes: $s_\theta \neq 0$.

**Normalized image plane:**
Given known intrinsic matrix $K$, it'd be convenient to work with normalized image coordinates $\hat x$, $\hat y$ by warping the image with:
$$\begin{bmatrix}
\hat x\\
\hat y\\
1
\end{bmatrix} = K^{-1} \begin{bmatrix}
x\\
y\\
1
\end{bmatrix}$$
The normalized image coordinates can be obtained by a virtual camera with $\hat K = I$.

(Aside) Duality:
The line: $2x + 4y + 1 = 0$ can be expressed as:
$$\begin{bmatrix} 2 \\ 4 \\ 1 \end{bmatrix}^T \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = 0$$
Therefore, the homogeneous coordinate $[2 \ 4 \ 1]^T$ can either be a point or a line.

To find the intersection of line $[2 \ 4\ 1]^T x = 0$ and $[3 \ 1\ 3]^T x = 0$: 
$$\begin{bmatrix} 2 \\ 4 \\ 1 \end{bmatrix} \times \begin{bmatrix} 3 \\ 1 \\ 3 \end{bmatrix}$$

# Homography
Homography is a bijective projective transformation $\mathbb{P}^2 \rightarrow \mathbb{P}^2$ that can be represented by an invertible $3 \times 3$ matrix up to scale.
## Planar Interpretation
The planar interpretation describes how a plane appears differently in two camera views.

![[homography-planar.png]]

Let's place the world coordinate frame on this plane. For the full camera projection:
$$\lambda \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} m_{11} & m_{12} & m_{13} & m_{14} \\ m_{21} & m_{22} & m_{23} & m_{24} \\ m_{31} & m_{32} & m_{33} & m_{34} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$
Plug in $Z = 0$:
$$\lambda \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} m_{11} & m_{12} & m_{14} \\ m_{21} & m_{22} & m_{24} \\ m_{31} & m_{32} & m_{34} \end{bmatrix} \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}$$
Therefore, **homography**:
$$\lambda \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = H \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}$$
And, **inverse homography**:
$$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = \lambda H^{-1} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

With homography and inverse homography, we can relate two camera views of the same plane:
$$\begin{bmatrix} x_2 \\ y_2 \\ 1 \end{bmatrix} \equiv H_2H_1^{-1} \begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix}$$
## Rotational Interpretation
The rotational interpretation describes the mapping between two images taken by two cameras with the same translation but different rotations.

![[homography-rotational.png]]

Relation between camera coordinates:
$$\begin{bmatrix} X_2 \\ Y_2 \\ Z_2 \end{bmatrix} = R \begin{bmatrix} X_1 \\ Y_1 \\ Z_1 \end{bmatrix}$$
Perspective projection:
$$\begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix} \equiv K_1 \begin{bmatrix} X_1 \\ Y_1 \\ Z_1 \end{bmatrix},\ \begin{bmatrix} x_2 \\ y_2 \\ 1 \end{bmatrix} \equiv K_2 \begin{bmatrix} X_2 \\ Y_2 \\ Z_2 \end{bmatrix}$$
Combining both:
$$\begin{bmatrix} x_2 \\ y_2 \\ 1 \end{bmatrix} \equiv K_2RK_1^{-1} \begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix}$$

(Aside) 360 images stored in 2D:
$$L(\theta, \phi) = [R, G, B]$$
## Homography Estimation
Given corresponding 2D points in two images, estimate homography matrix:
$$H = \begin{bmatrix}
a & b & c\\
d & e & f\\
g & h & i
\end{bmatrix}$$

**Direct Linear Transform (DLT)**:
Direct linear transform (DLT) reformulates the original problem into a linear system $Ah = 0$ and solves it vis **SVD**:
$$
x_2 = \frac{ax_1 + by_1 + c}{gx_1 + hy_1 + i}
$$
$$
x_2(gx_1 + hy_1 + i) = ax_1 + by_1 + c
$$
$$
\begin{bmatrix} -x_1 & -y_1 & -1 & 0 & 0 & 0 & x_1x_2 & y_1x_2 & x_2 \end{bmatrix} \begin{bmatrix} a \\ b \\ c \\ d \\ e \\ f \\ g \\ h \\ i \end{bmatrix} = 0
$$
Therefore:
$$
Ah = \begin{bmatrix} 0 \\ 0 \\ \vdots \end{bmatrix}
$$
- The DOF of $h$ is 8 (because $H$ and $\alpha H$ are the same homography matrix).
- Therefore, we need 4 pairs of corresponding points to solve for $h$.

You can use RANSAC when estimating homography.