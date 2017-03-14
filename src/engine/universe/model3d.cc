/*
* Model3d class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/model3d.h"

#include <SOIL/SOIL.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cilk/cilk.h>

#include "engine/helpers/proxy_config.h"

namespace engine {
namespace universe {

namespace {
    /* Unique id for object */
    static int objectid = 1;
}

/* Constructor */
Model3D::Model3D()
{
    id_ = objectid++;
    distance_ = -1.0f;
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

/* Compute coords and move for the model */
void Model3D::PrepareDraw(Camera *cam)
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
        element->ComputeMVP(cam);
    }
    distance_ = -1.0f;
    /* An object cant touch same object twice, except camera */
    if (!is_controlled_ && obstacle_ != nullptr)
        id_last_collision_ = obstacle_->id();
    obstacle_ = nullptr;
}

/* Draw the model */
void Model3D::Draw()
{
    for (auto &element : elements_) {
        element->Draw();
    }
}

/* Task who detect if obstacle is in collision with current object */
std::vector<Model3D*> Model3D::DetectCollision(Model3D *obstacle, tbb::mutex &collision_mutex,
                                               engine::parallell::EngineParallell *proxy_parallell)
{
    float distance = 1.0f;
    std::vector<float> distances;
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

    /* Compute distance collision thanks to parallell library (opencl, cilk, or no parallell use) */
    distance = proxy_parallell->ComputeCollisionParallell(box1, box2);

    /* Compute distance and update collision properties if needed */
    if (distance != 1.0f) {
        /* Protect conccurency update for objects in a room */
        tbb::mutex::scoped_lock lock_c(collision_mutex);
        if ((obstacle_ == nullptr || distance < distance_) &&
            (obstacle->obstacle() == nullptr || distance < obstacle->distance())) {
                /* Print debug if setting */
                using engine::helpers::ProxyConfig;
                if (ProxyConfig::getSetting<int>("debug") >= ProxyConfig::kDEBUG_COLLISION)
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
    }

    return recompute;
}

}//namespace geometry
}//namespace engine
