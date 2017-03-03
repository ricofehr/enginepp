#include <iostream>
#include "engine/universe/universe.h"
#include "engine/helpers/proxycl.h"
#include "engine/helpers/proxygl.h"

int main(int argc, char *argv[])
{
    using engine::universe::Universe;

	/* Init world */
//    engine::helpers::proxycl::InitCollisionCL();
    engine::helpers::proxygl::initGL();
    auto engine_universe{std::make_unique<Universe>()};
    engine::helpers::proxygl::settingsGL(engine_universe.get());
}
