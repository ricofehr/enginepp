#include <iostream>
#include <string>

#include "engine/universe/universe.h"
#include "engine/core/config_engine.h"
#include "engine/renderer/loopgl.h"

int main(int argc, char *argv[])
{
    using engine::universe::Universe;
    using engine::renderer::LoopGL;
    using engine::core::ConfigEngine;

    /* Init Config */
    ConfigEngine::InitConfig();

    /* Manage program parameters */
    ConfigEngine::ManageProgramParameters(argc, argv);

	/* Init world */
    LoopGL::Instance().InitGL();
    auto engine_universe{std::make_unique<Universe>()};
    LoopGL::Instance().Loop(engine_universe.get());
}
