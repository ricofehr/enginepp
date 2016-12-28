# enginepp

A 3d engine at very early stage and in working progress.
(C++ version of https://github.com/ricofehr/engine, with improved features)

The program generates 4 rooms with some bricks created and moved randomly,
The camera can move with mouse (head orientation) and arrow keys (camera direction).

## Compile

Needs OpenGL3 (>3.3) and GLew / GLM / SOIL / Glfw libraries.

```
$ cmake .
-- Configuring done
-- Generating done
-- Build files have been written to: ~/enginepp

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

- C++11
- Opengl 3
- Use of Glew, GLM, SOIL, Glfw libraries
- CMake for compile

## Folders
```
+--src/ Sources
+--bin/		Binary folder where engine executable is written
+--assets/      Texture files
```

## Run

Use mouse for head orientation and arrow keys for camera move.
When we cross a door, we change room (4 rooms).

```
bin/./engine
```

## Todo

- Improve collision algorithm
- Manage shadows and lights
- Improve camera move and interactions
- Manage drawing of more 3d models 
