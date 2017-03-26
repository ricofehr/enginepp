/*
* ProxyConfig Class File
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/helpers/proxy_config.h"

#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include "engine/parallell/engine_parallell.h"

namespace engine {
namespace helpers {

/*
*   Parse config file and set config values
*/
void ProxyConfig::ParseFile()
{
    /* Default or overriden setting file */
    struct stat buffer;
    std::string cfg_file = stat("config/enginepp.ini", &buffer) == 0 ? "config/enginepp.ini" : "config/enginepp.ini.default";

	/* Read the file. If there is an error, report it and exit. */
	try {
        cfg.readFile(cfg_file.c_str());
  	}
  	catch(const libconfig::FileIOException &fioex) {
    	std::cerr << "I/O error while reading file, config parser cancelled: " << cfg_file << std::endl;
  	}
  	catch(const libconfig::ParseException &pex) {
    	std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                  << " - " << pex.getError() << std::endl;
    	exit(EXIT_FAILURE);
	}
}

/*
*   Set default config values if defined neither into config file,
*   neither in program arguments.
*   Ensure all setting parameters are value.
*/
void ProxyConfig::DefaultValues()
{
    using engine::parallell::EngineParallell;

    if (!exists("parallell")) {
        setValue("parallell", libconfig::Setting::TypeInt, EngineParallell::kPARALLELL_SERIAL);
    }

    if (!exists("width")) {
        setValue("width", libconfig::Setting::TypeFloat, 800.0f);
    }

    if (!exists("height")) {
        setValue("height", libconfig::Setting::TypeFloat, 600.0f);
    }

    if (!exists("objects_count")) {
        setValue("objects_count", libconfig::Setting::TypeInt, 16);
    }

    if (!exists("load_objects_freq")) {
        setValue("load_objects_freq", libconfig::Setting::TypeFloat, 0.0f);
    }

    if (!exists("rooms_count")) {
        setValue("rooms_count", libconfig::Setting::TypeInt, 4);
    }

    if (!exists("granularity")) {
        setValue("granularity", libconfig::Setting::TypeInt, 16);
    }

    if (!exists("clipping")) {
        setValue("clipping", libconfig::Setting::TypeInt, 0);
    }

    if (!exists("vsync")) {
        setValue("vsync", libconfig::Setting::TypeBoolean, true);
    }

    if (!exists("grid")) {
        setValue("grid", libconfig::Setting::TypeBoolean, false);
    }

    if (!exists("debug")) {
        setValue("debug", libconfig::Setting::TypeInt, 0);
    }

    if (!exists("execution_time")) {
        setValue("execution_time", libconfig::Setting::TypeInt, 0);
    }
}

/* Parse and display config */
void ProxyConfig::ParseConfig() const
{
    std::cout << "Parallell mode (1 -> serial, 2 -> cilkplus, 3 -> opencl): " << getValue<int>("parallell") << std::endl;
    std::cout << "Window width: " << getValue<float>("width") << std::endl;
    std::cout << "Window height: " << getValue<float>("height") << std::endl;
    std::cout << "Objects count: " << getValue<int>("objects_count") << std::endl;
    std::cout << "Load objects sequentially in time: " << getValue<bool>("load_objects_seq") << std::endl;
    std::cout << "Rooms count: " << getValue<int>("rooms_count") << std::endl;
    std::cout << "Collision granularity: " << getValue<int>("granularity") << std::endl;
    std::cout << "Clipping (0 -> no clipping, 1 -> low clipping, 2 -> high clipping): " << getValue<int>("clipping") << std::endl;
    std::cout << "Execution Time (0 -> no limit): " << getValue<int>("execution_time") << std::endl;
    std::cout << "Vsync (limit framerate to monitor): " << getValue<bool>("vsync") << std::endl;
    std::cout << "Grid mode (not fill polygons): " << getValue<bool>("grid") << std::endl;
    std::cout << "Debug mode (0 -> no debug, 1 -> performance debug, 2 -> collision debug, 3 -> all debug): " << getValue<int>("debug") << std::endl;
}

/* Manage program parameters */
void ProxyConfig::ManageProgramParameters(int argc, char *argv[])
{
    using engine::parallell::EngineParallell;

    auto is_display_config = false;
    auto cnt = 1;
    while (cnt < argc) {
        /* Cant use switch with string */
        const std::string arg(argv[cnt++]);

        /* parallell type */
        if (arg == "-p") {
            const std::string arg2(argv[cnt++]);
            if (arg2 == "serial")
                ProxyConfig::setSetting("parallell", libconfig::Setting::TypeInt, EngineParallell::kPARALLELL_SERIAL);
            if (arg2 == "cilkplus")
                ProxyConfig::setSetting("parallell", libconfig::Setting::TypeInt, EngineParallell::kPARALLELL_CILK);
            if (arg2 == "opencl")
                ProxyConfig::setSetting("parallell", libconfig::Setting::TypeInt, EngineParallell::kPARALLELL_CL);
        }

        /* Count of objects in rooms */
        if (arg == "-o") {
            const std::string arg2(argv[cnt++]);
            ProxyConfig::setSetting("objects_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Count of rooms */
        if (arg == "-r") {
            const std::string arg2(argv[cnt++]);
            ProxyConfig::setSetting("rooms_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* granularity */
        if (arg == "-g") {
            const std::string arg2(argv[cnt++]);
            ProxyConfig::setSetting("granularity", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Debug mode */
        if (arg == "-d") {
            const std::string arg2(argv[cnt++]);
            ProxyConfig::setSetting("debug", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Clipping */
        if (arg == "-c") {
            const std::string arg2(argv[cnt++]);
            ProxyConfig::setSetting("clipping", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Execution Time */
        if (arg == "-e") {
            const std::string arg2(argv[cnt++]);
            ProxyConfig::setSetting("execution_time", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Help */
        if (arg == "-h") {
            std::cout << argv[0] << " can be used with following options who overrides config file" << std::endl;
            std::cout << "-c n  Clipping, 0: no clipping, 1: low clipping, 2: high clipping" << std::endl;
            std::cout << "-d n  Debug mode, 0: no debug, 1: performance debug, 2: collision debug, 3: all debug" << std::endl;
            std::cout << "-e n  Execution Time, 0: no limit" << std::endl;
            std::cout << "-g n  Granularity on collision computes" << std::endl;
            std::cout << "-h    Display help" << std::endl;
            std::cout << "-l    Display config" << std::endl;
            std::cout << "-o n  Count of objects in rooms" << std::endl;
            std::cout << "-p serial|cilkplus|opencl" << std::endl
                      << "      serial: no parallellism" << std::endl
                      << "      cilkplus: use intel cilkplus library" << std::endl
                      << "      opencl: intel cilkplus for all parallell computes but opencl for collision computes" << std::endl;
            std::cout << "-r n  Count of rooms" << std::endl;
            exit(0);
        }

        /* Display config values from file */
        if (arg == "-l") {
            is_display_config = true;
        }
    }

    /* Display config values from file */
    if (is_display_config) {
        ProxyConfig::DisplayConfig();
        exit(0);
    }
}

} //namespace helpers
} //namespace engine
