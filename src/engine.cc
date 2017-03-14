#include <iostream>
#include <string>

#include "engine/universe/universe.h"
#include "engine/helpers/proxy_config.h"
#include "engine/helpers/proxygl.h"

int main(int argc, char *argv[])
{
    using engine::universe::Universe;
    using engine::helpers::proxygl::InitGL;
    using engine::helpers::proxygl::SettingsGL;
    using engine::helpers::ProxyConfig;

    /* Init Config */
    ProxyConfig::InitConfig();

    /* Manage program parameters */
    ProxyConfig::ManageProgramParameters(argc, argv);

	/* Init world */
    InitGL();
    auto engine_universe{std::make_unique<Universe>()};
    SettingsGL(engine_universe.get());
}
