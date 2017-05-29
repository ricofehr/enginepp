/*
 * CollisionEngine class file: Implements the default DectectCollision function
 * @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 */

#include "engine/physics/collision_engine.h"

#include <iostream>
#include <string>
#include <vector>

#include "engine/universe/model3d.h"
#include "engine/core/config_engine.h"

namespace engine {
namespace physics {

/*
 *  Detect if obstacle is in collision with current object
 *  Return vector (empty by default) of Model3D who need a recompute for collision
 */
std::vector<engine::universe::Model3D*> CollisionEngine::DetectCollision(engine::universe::Model3D *object,
                                                                         engine::universe::Model3D *obstacle,
                                                                         tbb::mutex &collision_mutex)
{
    float distance = 1.0f;
    std::vector<float> distances;
    int fact = 1, fact2 = 0;
    std::vector<engine::universe::Model3D*> recompute;
    engine::universe::Model3D *oldobstacle1{nullptr}, *oldobstacle2{nullptr};

    engine::geometry::Box border1 = object->border();
    engine::geometry::Box border2 = obstacle->border();
    std::vector<glm::vec3> coords1 = border1.ComputeCoords();
    std::vector<glm::vec3> coords2 = border2.ComputeCoords();

    /* First polygon point (x,y,z) and dimensions (h,w,d) */
    GLfloat x1, y1, z1, h1, w1, d1;
    GLfloat x2, y2, z2, h2, w2, d2;
    GLfloat move1x, move1y, move1z, move2x, move2y, move2z;


    /* Cant touch same object twice in short time */
    if (obstacle->id() == object->id_last_collision()) {
        return recompute;
    }

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

    move1x = border1.move()[0];
    move1y = border1.move()[1];
    move1z = border1.move()[2];
    move2x = border2.move()[0];
    move2y = border2.move()[1];
    move2z = border2.move()[2];

    float box1[9] = {x1, y1, z1, w1, h1, d1, move1x, move1y, move1z};
    float box2[9] = {x2, y2, z2, w2, h2, d2, move2x, move2y, move2z};

    /* Compute distance collision thanks to parallell library (opencl, cilk, or no parallell use) */
    distance = ComputeCollision(box1, box2);

    /* Compute distance and update collision properties if needed */
    if (distance != 1.0f) {
        /* Protect conccurency update for objects in a room */
        tbb::mutex::scoped_lock lock_c(collision_mutex);
        if ((object->obstacle() == nullptr || distance < object->distance()) &&
            (obstacle->obstacle() == nullptr || distance < obstacle->distance())) {
            /* Print debug if setting */
            using engine::core::ConfigEngine;
            if (ConfigEngine::getSetting<int>("debug") >= ConfigEngine::kDEBUG_COLLISION) {
                std::cerr << "Obstacle::" << obstacle->id() << ", distance::" << distance << std::endl;
            }

            oldobstacle1 = object->obstacle();
            oldobstacle2 = obstacle->obstacle();

            object->set_obstacle(obstacle);
            object->set_distance(distance);
            obstacle->set_distance(distance);
            obstacle->set_obstacle(object);

            /* Recompute for polygons unbinded */
            if (oldobstacle1 != nullptr) {
                oldobstacle1->set_distance(-1);
                oldobstacle1->set_obstacle(nullptr);
                if (oldobstacle1->IsMoved()) {
                    recompute.push_back(oldobstacle1);
                }
            }

            if (oldobstacle2 != nullptr &&
                oldobstacle2->IsMoved() &&
                oldobstacle2->id() != object->id()) {
                oldobstacle2->set_distance(-1);
                oldobstacle2->set_obstacle(nullptr);
                recompute.push_back(oldobstacle2);
            }
        }
    }

    return recompute;
}

}//namespace helpers
}//namespace engine
