/*
 *  ConfigEngine Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 */

#include "engine/core/config_engine.h"

#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <cilk/cilk_api.h>

#include "engine/physics/collision_engine.h"

namespace engine {

namespace core {

void ConfigEngine::ParseFile()
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

void ConfigEngine::DefaultValues()
{
    using engine::physics::CollisionEngine;

    if (!exists("parallell")) {
        setValue("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_SERIAL);
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

    if (!exists("workers_count")) {
        setValue("workers_count", libconfig::Setting::TypeInt, 0);
    }
}

void ConfigEngine::ParseConfig() const
{
    auto nworkers = getValue<int>("workers_count") ? getValue<int>("workers_count") : __cilkrts_get_nworkers();

    std::cout << "Parallell mode (1 -> serial, 2 -> cilkplus, 3 -> opencl): " << getValue<int>("parallell") << std::endl;
    std::cout << "Window width: " << getValue<float>("width") << std::endl;
    std::cout << "Window height: " << getValue<float>("height") << std::endl;
    std::cout << "Objects count: " << getValue<int>("objects_count") << std::endl;
    std::cout << "Load objects frequency in time: " << getValue<float>("load_objects_freq") << std::endl;
    std::cout << "Rooms count: " << getValue<int>("rooms_count") << std::endl;
    std::cout << "Collision granularity: " << getValue<int>("granularity") << std::endl;
    std::cout << "Clipping (0 -> no clipping, 1 -> low clipping, 2 -> high clipping): " << getValue<int>("clipping") << std::endl;
    std::cout << "Workers count: " << nworkers << std::endl;
    std::cout << "Execution Time (0 -> no limit): " << getValue<int>("execution_time") << std::endl;
    std::cout << "Vsync (limit framerate to monitor): " << getValue<bool>("vsync") << std::endl;
    std::cout << "Grid mode (not fill polygons): " << getValue<bool>("grid") << std::endl;
    std::cout << "Debug mode (0 -> no debug, 1 -> test debug, 2 -> performance debug, 3 -> collision debug, 4 -> all debug): " << getValue<int>("debug") << std::endl;
}

void ConfigEngine::ManageProgramParameters(int argc, char* argv[])
{
    using engine::physics::CollisionEngine;

    auto is_display_config = false;
    auto cnt = 1;
    while (cnt < argc) {
        /* Cant use switch with string */
        const std::string arg(argv[cnt++]);

        /* Clipping */
        if (arg == "-c") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("clipping", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Debug mode */
        if (arg == "-d") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("debug", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Execution Time */
        if (arg == "-e") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("execution_time", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* granularity */
        if (arg == "-g") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("granularity", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Help */
        if (arg == "-h") {
            std::cout << argv[0] << " can be used with following options who overrides config file" << std::endl;
            std::cout << "-c n   Clipping, 0: no clipping, 1: low clipping, 2: high clipping" << std::endl;
            std::cout << "-d n   Debug mode, 0: no debug, 1: test debug, 2: performance debug, 3: collision debug, 4: all debug" << std::endl;
            std::cout << "-e n   Execution Time, 0: no limit" << std::endl;
            std::cout << "-g n   Granularity on collision computes" << std::endl;
            std::cout << "-h     Display help" << std::endl;
            std::cout << "-l     Display config" << std::endl;
            std::cout << "-o n   Count of objects in rooms" << std::endl;
            std::cout << "-p serial|cilkplus|opencl" << std::endl
            << "       serial: no parallellism" << std::endl
            << "       cilkplus: uses intel cilkplus library" << std::endl
            << "       opencl: uses opencl for collision computes" << std::endl;
            std::cout << "-r n   Count of rooms" << std::endl;
            std::cout << "-s n.m Load objects frequency, 0: generates all objects at start" << std::endl;
            std::cout << "-v 1|0 Enable/Disable vsync" << std::endl;
            std::cout << "-w n   Workers (cpu core) count (disabled if -p serial), 0: no limit, all cpu cores" << std::endl;
            exit(0);
        }

        /* Display config values from file */
        if (arg == "-l") {
            is_display_config = true;
        }

        /* Count of objects in rooms */
        if (arg == "-o") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("objects_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* parallell type */
        if (arg == "-p") {
            const std::string arg2(argv[cnt++]);
            if (arg2 == "serial") {
                ConfigEngine::setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_SERIAL);
            }

            if (arg2 == "cilkplus") {
                ConfigEngine::setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_CILK);
            }

            if (arg2 == "opencl") {
                ConfigEngine::setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_CL);
            }
        }

        /* Count of rooms */
        if (arg == "-r") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("rooms_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Sequentially load frequency */
        if (arg == "-s") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("load_objects_freq", libconfig::Setting::TypeFloat, std::stof(arg2));
        }

        /* Vsync */
        if (arg == "-v") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("vsync", libconfig::Setting::TypeBoolean, std::stoi(arg2) == 1);
        }

        /* Workers count */
        if (arg == "-w") {
            const std::string arg2(argv[cnt++]);
            ConfigEngine::setSetting("workers_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }
    }

    /* Display config values from file */
    if (is_display_config) {
        ConfigEngine::DisplayConfig();
        exit(0);
    }
}

} //namespace core

} //namespace engine
