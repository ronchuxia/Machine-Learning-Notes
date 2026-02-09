# `project`
Set the CMake variable `PROJECT_NAME`, `PROJECT_SOURCE_DIR`, etc.
```CMake
project(<project_name>)
```

# `find_package`
```CMake
find_package(<package_name> <version> <options>)
find_package(rclcpp REQUIRED)
```
`find_package` look outside your immediate project to find external libraries or software packages (like OpenCV, Qt, or Boost) so you can link them to your code.

When you call `find_package(LibName)`, CMake typically looks for a specific file (usually named `LibNameConfig.cmake` or `FindLibName.cmake`).

ROS packages use ament to export their information. When Package A is built, it installs a file named `PackageAConfig.cmake` into the `install/` folder. This is exactly what `find_package` looks for.

# `add_executable`
Create the program.
```CMake
add_executable(<target_name> <source_files>)
add_executable(talker src/talker.cpp)
```

# `target_link_libraries`
Define the relationship between your target (executable or library) and the dependencies it needs to run.
```CMake
target_link_library(<target_name> <keyword> <namespace::target>)
```

Keywords:
- `PRIVATE`: CMake adds the library to the compiler's link command for your project, but it does not add the library's header folders to the include path of people using your project.
- `PUBLIC`: CMake tells everyone: "If you use my library, you also need to add this other library's headers to your search path because my headers won't work without them."
- `INTERFACE`

# `install`
`install` copies only the necessary final files to a clean, organized `install/` directory where they can actually be run or used by other people.
