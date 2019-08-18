/**
 *  @file demo_level.cc
 *  @brief DemoLevel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/demo_level.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

std::unique_ptr<nextfloor::objects::Mesh> DemoLevel::GenerateUniverse() noexcept
{
    using nextfloor::core::CommonServices;
    auto universe = CommonServices::getFactory()->MakeUniverse();
    auto first_room_location = glm::vec3(0.0f);
    auto first_room = CommonServices::getFactory()->MakeRoom(first_room_location);
    auto room_dimension = first_room->dimension();
    first_room->add_child(CommonServices::getFactory()->MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
    first_room->add_child(CommonServices::getFactory()->MakeRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
    universe->add_child(std::move(first_room));

    /* create 26 more rooms around the first one */
    for (auto factor_x = -1.0f; factor_x <= 1.0f; factor_x++) {
        for (auto factor_y = -1.0f; factor_y <= 1.0f; factor_y++) {
            for (auto factor_z = -1.0f; factor_z <= 1.0f; factor_z++) {
                auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
                if (room_location != first_room_location) {
                    auto room = CommonServices::getFactory()->MakeRoom(room_location);
                    room->add_child(CommonServices::getFactory()->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
                    room->add_child(CommonServices::getFactory()->MakeRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
                    universe->add_child(std::move(room));
                }
            }
        }
    }

    universe->AddIntoChild(CommonServices::getFactory()->MakePlayer(glm::vec3(0.0f, -2.0f, 7.0f)));

    return universe;
}

} // namespace gameplay

} // namespace nextfloor
