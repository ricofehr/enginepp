#include <iostream>
#include <string>

#include "engine/universe/universe.h"
#include "engine/helpers/proxygl.h"
#include "engine/parallell/engine_parallell.h"

int main(int argc, char *argv[])
{
    using engine::universe::Universe;
    using engine::parallell::EngineParallell;

    int type_parallell = EngineParallell::kPARALLELL_SERIAL;

    if (argc >= 3) {
        std::string arg = argv[2];
        if (arg == "cilkplus")
            type_parallell = EngineParallell::kPARALLELL_CILK;
        if (arg == "opencl")
            type_parallell = EngineParallell::kPARALLELL_CL;
    }

	/* Init world */
    engine::helpers::proxygl::initGL();
    auto engine_universe{std::make_unique<Universe>(type_parallell)};
    engine::helpers::proxygl::settingsGL(engine_universe.get());
}
