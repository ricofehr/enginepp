/*
* Model3d class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

/* OpenCL 1.2 */
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include "engine/universe/model3d.h"

#include <glm/glm.hpp>
#include <SOIL/SOIL.h>

#ifdef __APPLE__
    #define __CL_ENABLE_EXCEPTIONS
    #include <OpenCL/cl.hpp>
#else
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
    #include <CL/cl2.hpp>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "engine/geometry/box.h"
#include "engine/geometry/shape3d.h"

namespace engine {
namespace universe {

namespace {
    /* Unique id for object */
    static int objectid = 1;
    /* OpenCL env */
    static cl::Kernel collision_kernel;
    static cl::CommandQueue collision_queue;
    static cl::Buffer bufferbox1;
    static cl::Buffer bufferbox2;
    static cl::Buffer bufferdist;

    void initCollisionCL() {
        try {
            /* Query for platforms */
            std::vector <cl::Platform> platforms;
            cl::Platform::get(&platforms);

            /* Get a list of devices on this platform */
            std::vector<cl::Device> devices;
            platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

            /* Create a context for the devices */
            cl::Context context(devices);

            /* Create a command-queue for the first device */
            collision_queue = cl::CommandQueue(context, devices[0]);

            /* Create the memory buffers */
            bufferbox1 = cl::Buffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float));
            bufferbox2 = cl::Buffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float));
            bufferdist = cl::Buffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(float));

            /* Read the program Source */
            std::ifstream source_file("cl/collision_kernel.cl");
            std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
            cl::Program::Sources source(1, std::make_pair(source_code.c_str(), source_code.length() + 1));

            /* Create the program from the source code */
            cl::Program program = cl::Program(context, source);

            /* Compile the program for the devices */
            program.build(devices);

            /* Create the kernel */
            collision_kernel = cl::Kernel(program, "collision");

        } catch(cl::Error error) {
            std::cout << error.what() << "(" << error.err() << ")" << std::endl;
            exit(1);
        }
    }
}

/* Constructor */
Model3D::Model3D()
{
    /* Init openCl env only once */
    if (objectid == 1)
           initCollisionCL();

    id_ = objectid++;
    distance_ = -1;
    obstacle_ = nullptr;
    id_last_collision_ = 0;
    is_controlled_ = false;
    is_crossed_ = false;
}

/* Comparaison operator */
bool operator==(const Model3D& o1, const Model3D& o2)
{
    return o1.id() == o2.id();
}

bool operator!=(const Model3D& o1, const Model3D& o2)
{
    return o1.id() != o2.id();
}

/* Draw the model */
void Model3D::Draw()
{
    /* if we can cross into object, make distance to 15 */
    if (obstacle_ != nullptr && obstacle_->IsCrossed()) {
        distance_ = 15.0f;
        obstacle_ = nullptr;
    }

    border_.set_distance(distance_);
    border_.MoveCoords();
    for (auto &element : elements_) {
        element->set_distance(distance_);
        element->Draw();
    }
    distance_ = -1;
    /* An object cant touch same object twice, except camera */
    if (!is_controlled_ && obstacle_ != nullptr)
        id_last_collision_ = obstacle_->id();
    obstacle_ = nullptr;
}

/* Task who detect if obstacle is in collision with current object */
std::vector<Model3D*> Model3D::DetectCollision(Model3D *obstacle)
{
    float distance = 1.0f, distances[10];
    int fact = 1, fact2 = 0;
    std::vector<Model3D*> recompute;
    Model3D *oldobstacle1, *oldobstacle2;

    engine::geometry::Box border2 = obstacle->border();
    std::vector<glm::vec3> coords1 = border_.ComputeCoords();
    std::vector<glm::vec3> coords2 = border2.ComputeCoords();

    /* First polygon point (x,y,z) and dimensions (h,w,d) */
    GLfloat x1, y1, z1, h1, w1, d1;
    GLfloat x2, y2, z2, h2, w2, d2;
    GLfloat move1x, move1y, move1z, move2x, move2y, move2z;


    /* Cant touch same object twice in short time */
    if (obstacle->id() == id_last_collision_)
        return recompute;

    x1 = coords1.at(0)[0];
    y1 = coords1.at(0)[1];
    z1 = coords1.at(0)[2];
    x2 = coords2.at(0)[0];
    y2 = coords2.at(0)[1];
    z2 = coords2.at(0)[2];

    h1 = coords1.at(3)[1] - y1;
    w1 = coords1.at(1)[0] - x1;
    d1 = coords1.at(4)[2] - z1;
    h2 = coords2.at(3)[1] - y2;
    w2 = coords2.at(1)[0] - x2;
    d2 = coords2.at(4)[2] - z2;

    move1x = border_.move()[0];
    move1y = border_.move()[1];
    move1z = border_.move()[2];
    move2x = border2.move()[0];
    move2y = border2.move()[1];
    move2z = border2.move()[2];

    float box1[9] = {x1, y1, z1, w1, h1, d1, move1x, move1y, move1z};
    float box2[9] = {x2, y2, z2, w2, h2, d2, move2x, move2y, move2z};

    /* Compute distance collision thanks to opencl */
    try {

        /* Copy the input data to the input buffer */
        collision_queue.enqueueWriteBuffer(bufferbox1, CL_TRUE, 0, 9 * sizeof(float), box1);
        collision_queue.enqueueWriteBuffer(bufferbox2, CL_TRUE, 0, 9 * sizeof(float), box2);

        /* Set kernel arguments */
        collision_kernel.setArg(0, bufferbox1);
        collision_kernel.setArg(1, bufferbox2);
        collision_kernel.setArg(2, bufferdist);

        /* Execute the kernel (10 workitem in 1 workgroup => compute distance for the 10 fact) */
        cl::NDRange global(10);
        cl::NDRange local(10);
        collision_queue.enqueueNDRangeKernel(collision_kernel, cl::NullRange, global, local);

        /* Copy the output data back to the host */
        collision_queue.enqueueReadBuffer(bufferdist, CL_TRUE, 0, 10 * sizeof(float), distances);

        /* Get the min distance of the 10 facts */
        distance = 1.0f;
        for (int i = 0; i < 10; i++) {
            if (distances[i] != 1.0f) {
                distance = distances[i];
                break;
            }
        }
    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        distance = 1.0f;
    }

    /* Compute distance and update collision properties if needed */
    if (distance != 1.0f &&
        (obstacle_ == nullptr || distance < distance_) &&
        (obstacle->obstacle() == nullptr || distance < obstacle->distance())) {

            std::cerr << "Obstacle::" << obstacle->id() << ", distance::" << distance << std::endl;
            oldobstacle1 = obstacle_;
            oldobstacle2 = obstacle->obstacle();

            obstacle_ = obstacle;
            distance_ = distance;
            obstacle_->set_distance(distance);
            obstacle_->set_obstacle(this);

            /* Recompute for polygons unbinded */
            if (oldobstacle1 != nullptr) {
                oldobstacle1->set_distance(-1);
                oldobstacle1->set_obstacle(nullptr);
                if (oldobstacle1->IsMoved())
                    recompute.push_back(std::move(oldobstacle1));
            }

            if (oldobstacle2 != nullptr && oldobstacle2->IsMoved()) {
                oldobstacle2->set_distance(-1);
                oldobstacle2->set_obstacle(nullptr);
                recompute.push_back(std::move(oldobstacle2));
            }

            oldobstacle1 = nullptr;
            oldobstacle2 = nullptr;
    }

    return recompute;
}

}//namespace geometry
}//namespace engine
