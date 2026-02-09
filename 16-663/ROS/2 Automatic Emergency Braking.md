# Environment
**Docker Compose**: Docker Compose is a tool that allows you to define and run multi-container applications.

After changing the `docker-compose.yml`, run `docker-compose up` again. This will automatically detect your edit, stop the old container and start a new one.

**X11 (X Window System)**: X11 is a software that allows Unix-like operating systems to display a Graphical User Interface (GUI). 

X11 is a **client-server** architecture that runs as a separate layer.
- **The X Server:** This is the software running on the machine with the **physical screen and keyboard**. It manages the hardware and draws the pixels.
- **The X Client:** These are the **applications** (like Firefox or a Terminal). They send requests to the Server.
Because of this separation, you can run a Client application on a remote computer and have it display its window on your local Server (your laptop).

**VNC (Virtual Network Computing)**: VNC is a remote desktop sharing system based on the RFB (Remote Framebuffer) protocol.

When using VNC with X11, a special version of the X server creates a virtual screen in the remote computer's memory. VNC scrapes the pixels from that virtual screen, compresses them, and sends them over the network to your local computer.

**noVNC**: noVNC is an open-source VNC client written in JavaScript.

**X11 Forwarding**: SSH allows X11 forwarding. It sends **drawing commands** to your local computer.

# Vehicle Dynamics
**Ackermann Steering**: Ackermann steering is a geometric arrangement of linkages in a vehicle's steering system designed to solve a simple but critical problem: when a car turns, the inside wheel follows a tighter curve than the outside wheel.
![[Ackermann Steering.png]]

Velocity:
- **Longitudinal Velocity**
- **Lateral Velocity**

**Kinematic Bicycle Model**
![[Kinematic Bicycle Model.png]]

**Lateral Slip**: In vehicle dynamics, lateral slip (or more specifically, the slip angle) is the difference between the direction a tire is pointing and the direction it is actually traveling. 

Unlike a metal wheel on a train track, a rubber tire is flexible. When you turn the steering wheel, the tire doesn't just instantly change the vehicle's path; the rubber twists and deforms, creating a small mismatch between the wheel's orientation and its actual path over the ground.

# ROS Coordinate Transformation
**REP: ROS Enhancement Proposal**

**REP-105** defines the coordinate transformation between different coordinate frames. It mandates the standard **Tree** structure:
`earth` -> `map` -> `odom` -> `base_link`
- `earth`: The global coordinate system (like Latitude/Longitude/Altitude). This is used when you have multiple robots.
- `map`: A world-fixed frame. This is the origin of your specific 2D or 3D map.
- `odom`: A local-world frame. It is where the robot thinks it is based on its own internal sensors (wheel encoders or inertial measurement unit, IMU).
- `base_link`: The coordinate origin is physically attached to the car (usually the center of the rear axle).

The `odom` can be calculated by comparing two Lidar measurements.

**REP-103** defines the **physical units** and the **orientation of the axes**.
- Units: always use meters, radians and seconds
- Axis: right hand rule (x forward, y left, z up)
- Rotation: counter-clockwise is positive

We can use TF2 to inspect coordinate frames and coordinate transformations.
```bash
ros2 run tf2_tools view_frames
```


