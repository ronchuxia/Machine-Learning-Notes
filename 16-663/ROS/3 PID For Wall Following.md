# Visualization
You can publish messages to a debug topic and use `rqt_plot` or PlotJuggler to visualize the debug data.

# Smoothing the Derivative
Due to very small dt, the derivative term is very noisy. You can:
1. Filter the derivative term (e.g. EMA).
2. Compute an average of the error over a window of measurements.
3. Use a larger dt (update the derivative term every several iterations).

# Creating Custom Messages
https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.htm

# Vehicle Setup
## Network
**IP**
These settings define your device's identity within its immediate network.
- **IP Address (of your device):** This is your device’s specific address.
- **Subnet Mask (of the local subnet):** This acts as a filter. It tells your computer which part of the IP address represents the **Network** and which part represents the **Host**.
- **Gateway:** This is the exit door of your local network, usually your router's internal IP address. If your computer wants to send data to an address that isn't on its local street, it sends it to the Gateway to be forwarded to the outside world (the Internet).

**Route**
While Address settings tell you who you are, Route settings tell the device how to get to other places.
- **IP Address (of the destination network):** Instead of a single device, this usually refers to an entire destination network (e.g., `10.0.0.0`) that you want to reach.
- **Subnet Mask (in routing):** This defines the size of the destination network. A route to `172.16.0.0` with a mask of `255.255.0.0` tells the router that any IP starting with `172.16` should follow this specific path.
- **Gateway (next hop):** In routing, the gateway is the **next router** in the chain.
    - If you are trying to reach a server across the country, your local routing table says: "To get to Network X, send the data to Gateway Y."
    - Gateway Y then looks at its routing table and sends it to Gateway Z, and so on.

In most home or small office setups, you rarely need to touch the routing table because the default gateway (your router) handles everything automatically.
## Device
**udev**: udev is the **device manager** for the Linux kernel.
- **Dynamic Node Creation:** It creates or removes device files (found in the `/dev` directory).
- **Persistent Naming:** It ensures your devices get the same name every time.
- **User-Space Actions:** It can trigger specific scripts or programs when a device is connected.

udev operates based on **rules files** located in `/etc/udev/rules.d/`.

`/etc/udev/ruls.d/99-hokuyo.rules`:
```
KERNEL=="ttyACM[0-9]*", ACTION=="add", ATTRS{idVendor}=="15d1", MODE="0666", GROUP="dialout", SYMLINK+="sensors/hokuyo"
```