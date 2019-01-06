# nextfloor

A 3d engine at very early stage and in working progress.
(C++ version of https://github.com/ricofehr/engine, with improved features)

The program generates randomly rooms with some randomly moving objects.
The camera can move with mouse (head orientation) and arrow keys (camera direction).

## Prerequisites

Needs Cmake (>3.1), OpenCL (1.2), TBB, OpenGL3 (>3.3) and GLew / GLM / SOIL / Glfw libraries.

On ubuntu or Debian, apt-get make most of prerequisites install
```
# apt-get install cmake make g++ libx11-dev libgl1-mesa-dev libglu1-mesa-dev 
# apt-get install libxrandr-dev libxext-dev libglfw3-dev libsoil-dev libglm-dev
# apt-get install libglew-dev opencl-headers libtbb-dev lsb-core libconfig++-dev
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

On OSX, we need XCode and install some libraries with brew (SOIL must be install manually)
```
$ brew install cmake glm glew glfw3 tbb libconfig
```

And SOIL must be install manually
```
$ git clone https://github.com/childhood/libSOIL
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

## Compile

In build directory, generate MakeFiles and compile project
```
$ cd build && cmake ../
-- Configuring done
-- Generating done
-- Build files have been written to:

$ make
Scanning dependencies of target nextfloor
[  5%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/graphics/shape3d.cc.o
[ 10%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/graphics/quad.cc.o
[ 15%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/graphics/cube.cc.o
[ 21%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/graphics/border.cc.o
[ 26%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/core/config_engine.cc.o
[ 31%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/renderer/loopgl.cc.o
[ 36%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/physics/collision_engine.cc.o
[ 42%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/physics/cl_collision_engine.cc.o
[ 47%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/physics/cilk_collision_engine.cc.o
[ 52%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/physics/serial_collision_engine.cc.o
[ 57%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/universe/random_universe_factory.cc.o
[ 63%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/universe/camera.cc.o
[ 68%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/universe/model3d.cc.o
[ 73%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/universe/wall.cc.o
[ 78%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/universe/brick.cc.o
[ 84%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/universe/room.cc.o
[ 89%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor/universe/universe.cc.o
[ 94%] Building CXX object CMakeFiles/nextfloor.dir/src/nextfloor.cc.o
[100%] Linking CXX executable ../bin/nextfloor
[100%] Built target nextfloor
```

## Features

- C++14
- OpenCL 1.2
- TBB
- Opengl 3
- Use of Glew, GLM, SOIL, Glfw libraries
- CMake for compile

## Folders
```
+--assets/  Texture files
+--bin/		Binary folder where nextfloor executable is written
+--build/   Build folder for compile stuffs
+--cl/      OpenCL Kernels folder
+--cmake/   Cmake modules folder
+--config/  Config folder
+--dia/     Uml and span diagrams
+--doxy/    Destination folder for documentation generated
+--glsl/    OpenGL Shaders folder
+--others/	Demo gif and others exported files
+--scripts/ Bash scripts
+--src/ 	Sources
+--tmp/     Temporary folder
```

## Settings

A default setting file is present at config/nextfloor.ini.default
For local config change, we can duplicate this file
```
$ cp config/nextfloor.ini.default config/nextfloor.ini
$ vi config/nextfloor.ini
```

It's also possible to change mostly setting on the fly with program parameters (See below).

## Run

Use mouse for head orientation and arrow keys for camera move.
And following keys are used
- 'p': pause / unpause
- 'esc': exit

Default run, without any parameter
```
bin/./nextfloor  # Use settings as setted in config file (config/nextfloor.ini or config/nextfloor.ini.default).
```

Program accept options who can override config settings
```
-c n   Clipping, 0: no clipping, 1: high clipping, 2: low clipping
-d n   Debug mode, 0: no debug, 1: test debug, 2: performance debug, 3: collision debug, 4: all debug
-e n   Execution Time, 0: no limit
-g n   Granularity on collision computes
-h     Display help
-l     Display config
-o n   Count of objects in rooms
-p serial|tbb|opencl
       serial: no parallellism
       tvv: uses intel tbb library
       opencl: uses opencl for collision computes
-r n   Count of rooms
-s n.m Load objects frequency, 0: generates all objects at start
-v 1|0 Enable/Disable vsync
-w n   Workers (cpu core) count (disabled if -p serial), 0: no limit, all cpu cores
```

For example
```
./bin/./nextfloor -d 0 -p tbb -o 48 -r 16 # no debug, cilkplus parallellism, 48 objects, 16 rooms
```

[![NextFloor](https://img.youtube.com/vi/7XnjXwFhHZk/0.jpg)](https://www.youtube.com/watch?v=7XnjXwFhHZk)

## Test

A bash script can be used for performance test
```
Usage: ./scripts/./test_perf.sh [options]

-h          Some help text.
-e path     Report folder (default ./tmp/)
-g n        Set granularity for collision computes (default is 64)
-nocl       Disable tests using opencl for parallell collision computes
-o n        Set count of moving objects into rooms (default is 32)
-r n        Set count of rooms (default is 4)
-t n        Select test(s) to execute (0 => no seq load and during 60s, 1 => seq load, 2 => both tests)
-w n        Set workers (cpu core) number (default is 0)
```

## Documentation

For generate doxygen documentation, into the project folder
```
doxygen doxygen.config
```

Doxy pages are available online [here](http://oxy.nextfloor.nextdeploy.io)

## UML

5 namespaces
- Core
- Graphics
- Physics
- Renderer
- Universe

Uses of Design Patterns
- Composite (Model3D / Universe / Room / Brick / Wall / Camera)
- Abstract Factory (UniverseFactory / RandomUniverseFactory)
- Strategy (CollisionEngine and his inherited classes)
- Singleton (ConfigEngine, LoopGL)

![Class Diagram](https://github.com/ricofehr/nextfloor/raw/master/others/classes.png)

## Parallell Computes

Below, the nextfloor span graph (v0.2)
![Span Diagram](https://github.com/ricofehr/nextfloor/raw/master/others/parallell_span.png)

During the tests, NextFloor (v0.2) is started with the following parameters
- granularity: 128
- vsync: false
- clipping: disabled
- 4 / 8 / 12 rooms
- 128 / 512 / 1152 Moving objects

And hardware used during test
- cpu: Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz (6 physical cores, 12 with intel ht)
- ram: 32Go at 2100Mhz
- cg: GeForce GTX 980 Ti (22 computes units)

#### 4 Rooms and 128 Moving Objects
![Test Result With 128 Moving Objects](https://github.com/ricofehr/nextfloor/raw/master/others/fpsbycores_128.png)
- optimal number of workers: 6
- speedup to 6 workers: 2
- efficiency to 6 workers: 33%

#### 8 Rooms and 512 Moving Objects
![Test Result With 512 Moving Objects](https://github.com/ricofehr/nextfloor/raw/master/others/fpsbycores_512.png)
- optimal number of workers: 8
- speedup to 8 workers: 3
- efficiency to 8 workers: 37%

#### 12 Rooms and 1152 Moving Objects
![Test Result With 1152 Moving Objects](https://github.com/ricofehr/nextfloor/raw/master/others/fpsbycores_1152.png)
- optimal number of workers: 12
- speedup to 12 workers: 4,2
- efficiency to 12 workers: 35%

We deduce that more moving objects there are, more the number of cores gives an increase in performance.

And going through opencl for collision calculation, the results provide an improvement only if the precision for the collisions is very very important.
Thus, with a granularity of 32768, one obtains a speedup of 2 and a efficiency of 10%.

It is concluded that the CPU parallelism of tbb brings real gains on NextFloor.
On the other hand, OpenCL is only adapted from a high level of calculations sent to the kernel on the GPU.
