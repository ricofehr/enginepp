/**
 *  @file collision_engine.cc
 *  @brief CollisionEngine class file.
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/collision_engine.h"

#include <iostream>
#include <vector>

#include "nextfloor/graphics/border.h"
#include "nextfloor/universe/model3d.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

void CollisionEngine::DetectCollision(nextfloor::universe::Model3D* target,
                                      nextfloor::universe::Model3D* obstacle)
{
    using nextfloor::universe::Model3D;
    using nextfloor::graphics::Border;

    float distance = 1.0f;

    const Border* border1 = target->border();
    const Border* border2 = obstacle->border();
    std::vector<glm::vec3> coords1 = border1->GetCoordsModelMatrixComputed();
    std::vector<glm::vec3> coords2 = border2->GetCoordsModelMatrixComputed();

    /* First polygon point (x,y,z) and dimensions (h,w,d) */
    GLfloat x1, y1, z1, h1, w1, d1;
    GLfloat x2, y2, z2, h2, w2, d2;
    GLfloat move1x, move1y, move1z, move2x, move2y, move2z;

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

    move1x = border1->move()[0];
    move1y = border1->move()[1];
    move1z = border1->move()[2];
    move2x = border2->move()[0];
    move2y = border2->move()[1];
    move2z = border2->move()[2];

    /* Prepare arrays for computecollision */
    float box1[9] = {x1, y1, z1, w1, h1, d1, move1x, move1y, move1z};
    float box2[9] = {x2, y2, z2, w2, h2, d2, move2x, move2y, move2z};

    /* If the both objects are already associated, return */
    if (target->obstacle() == obstacle) {
        return;
    }

    /* Compute distance collision thanks to parallell library (opencl, cilk, or no parallell use) */
    distance = ComputeCollision(box1, box2);
    if (distance == 1.0f) {
        return;
    }

    /* Update obstacle and distance if lower than former */
    target->lock();
    if (distance < target->distance()) {
        target->set_obstacle(obstacle);
        target->set_distance(-distance);

        /* Print debug if setting */
        using nextfloor::core::CommonServices;
        if (CommonServices::getConfig()->getDebugLevel() >= CommonServices::getLog()->kDEBUG_COLLISION) {
            std::cerr << "Object::" << target->id() << " - Obstacle::" << obstacle->id() << " - Distance::" << distance << std::endl;
        }
    }
    target->unlock();
}

} // namespace helpers

} // namespace nextfloor
