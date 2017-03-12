# enginepp

A 3d engine at very early stage and in working progress.
(C++ version of https://github.com/ricofehr/engine, with improved features)

The program generates 4 rooms with some bricks created and moved randomly,
The camera can move with mouse (head orientation) and arrow keys (camera direction).

## Prerequisites

Needs Cmake (>3.1), OpenCL (1.2), CilkPlus and TBB, OpenGL3 (>3.3) and GLew / GLM / SOIL / Glfw libraries.

On ubuntu or Debian, apt-get make most of prerequisites install
```
# apt-get install cmake make g++ libx11-dev libgl1-mesa-dev libglu1-mesa-dev libxrandr-dev libxext-dev libglfw3-dev libsoil-dev libglm-dev libglew-dev opencl-headers libtbb-dev lsb-core
```

For OpenCL, we need gpu library, for intel gpu
```
# apt-get install beignet-dev
```

Or for nvidia gpu
```
# apt-get install nvidia-opencl-dev
```

And intel (for intel cpu) runtime
```
$ wget "http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz"
$ tar xvfz opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz
$ cd  opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25 && ./install.sh
```

For CilkPlus installation, you can execute script below (detailed instructions on the [github page](https://github.com/cilkplus/cilkplus.github.com/blob/master/index.md#try-cilk-plusllvm)).
```
./scripts/./cilk_linux.sh
```

On OSX, we need XCode and install some libraries with brew (SOIL must be install manually)
```
$ brew install cmake glm glew glfw3 tbb
```

And SOIL must be install manually
```
$ git clone https://github.com/smibarber/libSOIL
$ cd libSOIL && make
$ sudo mkdir -p /usr/local/include/SOIL
$ sudo cp *.h /usr/local/include/SOIL/
$ sudo cp libSOIL.* /usr/local/lib/
```

And OpenCL must be install manually
```
$ wget https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp
$ sudo mv cl2.hpp /System/Library/Frameworks/OpenCL.framework/Headers/
```

For OpenCL header copy, need execute this in terminal on Recovery Mode
```
$ csrutil disable
```

For CilkPlus installation, you can execute script below (detailed instructions on the [github page](https://github.com/cilkplus/cilkplus.github.com/blob/master/index.md#try-cilk-plusllvm)).
```
./scripts/./cilk_osx.sh
```

## Compile

Step1, generate MakeFiles and check prerequisites
```
$ cmake .
-- Configuring done
-- Generating done
-- Build files have been written to: ~/enginepp
```

Step2, Before each compile, we need init env for cilkplus use.
On Linux
```
source ./scripts/./cilk_vars_linux.sh
```
On MacOS
```
source ./scripts/./cilk_vars_osx.sh
```

Step3, compile program
```
$ make
Scanning dependencies of target engine
[  6%] Building CXX object CMakeFiles/engine.dir/src/engine/geometry/shape3d.cc.o
[ 13%] Building CXX object CMakeFiles/engine.dir/src/engine/geometry/quad.cc.o
[ 20%] Building CXX object CMakeFiles/engine.dir/src/engine/geometry/cube.cc.o
[ 26%] Building CXX object CMakeFiles/engine.dir/src/engine/geometry/box.cc.o
[ 33%] Building CXX object CMakeFiles/engine.dir/src/engine/helpers/proxygl.cc.o
[ 40%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/camera.cc.o
[ 46%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/model3d.cc.o
[ 53%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/wall.cc.o
[ 60%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/window_model.cc.o
[ 66%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/door.cc.o
[ 73%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/brick.cc.o
[ 80%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/room.cc.o
[ 86%] Building CXX object CMakeFiles/engine.dir/src/engine/universe/universe.cc.o
[ 93%] Building CXX object CMakeFiles/engine.dir/src/engine.cc.o
[100%] Linking CXX executable bin/engine
[100%] Built target engine
```

## Features

- C++14
- OpenCL 1.2
- CilkPlus / TBB
- Opengl 3
- Use of Glew, GLM, SOIL, Glfw libraries
- CMake for compile

## Folders
```
+--src/ 	Sources
+--cl/      OpenCL Kernels folder
+--glsl/    OpenGL Shaders folder
+--bin/		Binary folder where engine executable is written
+--assets/  Texture files
+--demos/	Demo gif animated files
```

## Run

Use mouse for head orientation and arrow keys for camera move.
When we cross a door, we change room (4 rooms).

```
bin/./engine  # no parallell support: serial collision computes and unique core for other cilkplus loops
bin/./engine -p cilkplus # use cilkplus for collisions computes and other parallell loops
bin/./engine -p opencl # use opencl for collisions computes and cilkplus for other parallell loops
```
![Engine](demos/enginepp.gif?raw=true)

## Documentation

Doxy pages are available [here](http://oxy.enginepp.nextdeploy.io)

## UML

![Class Diagram](http://oxy.enginepp.nextdeploy.io/enginepp.png)

## Todo

- Improve collision algorithm
- Manage shadows and lights
- Improve camera move and interactions
- Manage drawing of more 3d models 
