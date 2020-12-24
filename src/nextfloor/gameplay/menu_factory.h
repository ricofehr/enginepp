/**
 *  @file menu_factory.h
 *  @brief Abstract Factory Class Implementation for game menus
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_MENUFACTORY_H_
#define NEXTFLOOR_GAMEPLAY_MENUFACTORY_H_

#include <memory>

#include "nextfloor/gameplay/menu.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class MenuFactory
 *  @brief Abstract Factory for game menus
 */
class MenuFactory {

public:
    virtual ~MenuFactory() = default;

    virtual std::unique_ptr<Menu> MakeMainMenu() const = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_MENUFACTORY_H_
