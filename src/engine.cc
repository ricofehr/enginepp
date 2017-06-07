#include <memory>

#include "engine/universe/random_universe_factory.h"
#include "engine/core/config_engine.h"
#include "engine/renderer/loopgl.h"

int main(int argc, char* argv[])
{
    using engine::universe::Universe;
    using engine::universe::RandomUniverseFactory;
    using engine::renderer::LoopGL;
    using engine::core::ConfigEngine;

    /* Reset seed */
    srand (time(NULL));

    /* Init Config */
    ConfigEngine::InitConfig();

    /* Manage program parameters */
    ConfigEngine::ManageProgramParameters(argc, argv);

	/* Init world */
    RandomUniverseFactory factory;
    LoopGL::Instance()->InitGL();
    factory.GenerateBuffers();
    std::unique_ptr<Universe> universe{factory.GenerateUniverse()};

    /* Launch GL Scene */
    LoopGL::Instance()->Loop(universe.get());
}
