# Source the Environment
```bash
source /opt/ros/foxy/setup.bash
```

# Creating a Package
Package creation in ROS 2 uses **ament** as its build system and **colcon** as its build tool.

**Build system**: A build system is the overall framework or methodology that defines how software is built from source code.
Examples: CMake (C/C++), PEP (Python), Colcon (ROS 2)

**Build tool**: A build tool is the program that actually executes the build according to the build system's instructions.
Examples: Make (CMake), Ninja (CMake), setuptools (Python), Colcon (ROS 2)

**Build type**: Packages in ROS 2 may be C++ or Python. Build type determines which internal build tool to use for this package.
- `ament_cmake`
- `ament_python`

Creating package:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake --license Apache-2.0 <package_name>
```
This will automatically create the CMakeLists.txt for CMake and package.xml for Colcon.

Creating a package of mixed C++/Python code: https://docs.ros.org/en/kilted/How-To-Guides/Ament-CMake-Python-Documentation.html

# Managing Dependency
`rosdep` is a dependency management utility that can work with packages and external libraries. 

`rosdep` is not a package manager in its own right; it is a meta-package manager that uses its own knowledge of the system and the dependencies to find the appropriate package to install on a particular platform. The actual installation is done using the system package manager (e.g. `apt` on Debian/Ubuntu, `dnf` on Fedora/RHEL, etc).

The `package.xml` is the file in your software where `rosdep` finds the set of dependencies.

Refer to https://docs.ros.org/en/kilted/Tutorials/Intermediate/Rosdep.html for the tags in the `package.xml`.

Declaring dependency:
```xml
<depend>rclcpp</depend>
<depend>ackermann_msgs</depend>
```
[[ackermann_msgs]]

Installing dependency:
```bash
cd ~/ros2_ws
rosdep update --include-eol-distros
rosdep install -i --from-path src --rosdistro foxy -y
```

# Client Libraries
**Client Libraries**: Client libraries are the APIs that allow users to implement their ROS 2 code. Using client libraries, users gain access to ROS 2 concepts such as nodes, topics, services, etc.
- `rclcpp`
- `rclpy`

```cpp
#include "rclcpp/rclcpp.hpp"
```

```python
import rclpy
```

# Node
**Nodes**: A node is a participant in the ROS 2 graph, which uses a client library to communicate with other nodes.

```cpp
class Talker : public rclcpp::Node
{
	# node
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Talker>());
    rclcpp::shutdown();
    return 0;
}
```

```python
from rclpy import Node

class Talker(Node):
	pass
	
def main(args=None):
    rclpy.init(args=args)
    talker = Talker()
    rclpy.spin(talker)
```

# Publisher and Subscriber
**Topics**: A topic is a communication bus used for the exchange of data between different parts. It follows a **Publisher/Subscriber** model.

**Publisher**
```cpp
publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("drive", 10);

auto msg = ackermann_msgs::msg::AckermannDriveStamped();
msg.drive.speed = speed;
publisher_->publish(msg);
```

```python
self.publisher = self.create_publisher(
	AckermannDriveStamped, # message type
	"drive", # topic name
	10 # history depth (QoS profile)
)

msg = AckermannDriveStamped()
msg.drive.speed = speed
self.publisher.publish(msg)
```

**Subscriber**
```cpp
auto drive_callback = [this](ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) -> void {
	// handles the msg
};

subscription_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>("drive", 10, drive_callback);
```

```python
def drive_callback(self, msg):
	pass

self.subscription = self.create_subscription(
	AckermannDriveStamped,
	'drive',
	self.drive_callback,
	10
)
```

In ROS2, the default QoS (Quality of Service) profile uses **Keep Last**.

NOTE: In ROS 2, messages aren't consumed or deleted once a subscriber reads them. Every subscriber tuned in to that topic receives a complete and identical copy of the message.

# VSCode Settings
`source /opt/ros/<distro>/setup.bash` sets the `$PYTHONPATH` environment variable. However, when you use VSCode remote development to attach to the container, this environment variable isn't set. You can manually add the python path to vscode python path by editing the `.vscode/settings.json`.

Note: the behavior is different for different ROS distributions.

For foxy:
```json
{
    "python.analysis.extraPaths": [
        "/opt/ros/foxy/lib/python3.8/site-packages"
    ],
    "python.autoComplete.extraPaths": [
        "/opt/ros/foxy/lib/python3.8/site-packages"
    ]
}
```

For humble:
```json
{
    "python.analysis.extraPaths": [
        "/opt/ros/humble/lib/python3.10/site-packages",
        "/opt/ros/humble/local/lib/python3.10/dist-packages"
    ],
    "python.autoComplete.extraPaths": [
        "/opt/ros/humble/lib/python3.10/site-packages",
        "/opt/ros/humble/local/lib/python3.10/dist-packages"
    ]
}
```

# Parameters
**Parameters**: Parameters in ROS 2 are associated with individual nodes. Parameters are used to configure nodes at startup (and during runtime), without changing the code.
## Declaring Parameters
```cpp
this->declare_parameter("v", 0.0); // Declares a parameter of type double
```

```python
self.declare_parameter("v", 0.0)
```

The parameter type is inferred from the default value.
## Getting Parameters
```cpp
double v = this->get_parameter("v").as_double();
```

```python
v = self.get_parameter("v").value
```
## Setting Parameters
List parameters.
```bash
ros2 param list
```

Set parameters via console.
```bash
ros2 param set /talker v 0.1
```

# Timer
```cpp
#include <chrono>
using namespace std::chrono_literals;

timer_ = this->create_wall_timer(10ms, std::bind(&Talker::timer_callback, this));
```

`using namespace std::chrono_literals;` allows you to use **time units** (like `s`, `ms`, `ns`) directly as suffixes on numbers.

```python
self.timer = self.create_timer(0.01, self.timer_callback) # 0.01s
```

# CMake
Find dependencies.
```cmake
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ackermann_msgs REQUIRED)
```

To support mixed C++/Python code.
```cmake
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})
```

Add executable and install to `lib/`.
```cmake
add_executable(talker src/talker.cpp)
ament_target_dependencies(talker rclcpp ackermann_msgs)
install(TARGETS talker DESTINATION lib/${PROJECT_NAME})
```
Install the executable to `lib/` so `ros2 run` can find it.

Install Python modules.
```CMake
install(
	PROGRAMS lab1_pkg/talker.py lab1_pkg/relay.py
	DESTINATION lib/${PROJECT_NAME}
)
```
Copy the Python module files to `lib/` so `ros2 run` can find it.

# Building Packages
After finishing write code, we build packages. 
```bash
cd ros2_ws
colcon build
```
Don't forget to source the environment before building.

# Running Packages
Source the setup file.
```bash
. install/setup.bash
```
`.` is a shell command that is an alias for the word `source`.

List packages.
```bash
ros2 pkg list
```

Run package.
```bash
ros2 run <package_name> <executable_name>
ros2 run safety_node safety_node # cpp
ros2 run safety_node safety_node.py # python
```

# Launch File
https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-CPP.html

# CLI Commands
List topic.
```bash
ros2 topic list
ros2 topic info /drive
```

Inspect topic live stream.
```bash
ros2 topic echo /drive
```

List node.
```bash
ros2 node list
ros2 node info /talker
```

Inspect message definition.
```bash
ros2 interface show geometry_msgs/msg/Twist
```

The message definition file can also be found at `/opt/ros/<distro>/share/` (e.g. `/opt/ros/<distro>/share/sensor_msgs/msg/LaserScan.msg`).

Publish a message to a topic.
```bash
ros2 topic pub /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
```





