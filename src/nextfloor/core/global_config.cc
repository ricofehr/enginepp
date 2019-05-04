/**
 *  @file global_config.cc
 *  @brief GlobalConfig Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/global_config.h"

#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <tbb/tbb.h>
#include <cassert>

#include "nextfloor/physics/collision_engine.h"

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

GlobalConfig::GlobalConfig()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void GlobalConfig::ParseConfigFile()
{
    struct stat buffer;
    std::string config_file = stat("config/nextfloor.ini", &buffer) == 0 ? "config/nextfloor.ini" : "config/nextfloor.ini.default";

    try {
        config.readFile(config_file.c_str());
    }
    catch(const libconfig::FileIOException &file_io_exception) {
        std::cerr << "I/O error while reading file, config parser cancelled: " << config_file << std::endl;
    }
    catch(const libconfig::ParseException &parse_exception) {
        std::cerr << "Parse error at " << parse_exception.getFile() << ":" << parse_exception.getLine()
        << " - " << parse_exception.getError() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void GlobalConfig::InitDefaultValues()
{
    using nextfloor::physics::CollisionEngine;

    if (!IsExist("parallell")) {
        setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_SERIAL);
    }

    if (!IsExist("width")) {
        setSetting("width", libconfig::Setting::TypeFloat, 800.0f);
    }

    if (!IsExist("height")) {
        setSetting("height", libconfig::Setting::TypeFloat, 600.0f);
    }

    if (!IsExist("objects_count")) {
        setSetting("objects_count", libconfig::Setting::TypeInt, 16);
    }

    if (!IsExist("load_objects_freq")) {
        setSetting("load_objects_freq", libconfig::Setting::TypeFloat, 0.0f);
    }

    if (!IsExist("rooms_count")) {
        setSetting("rooms_count", libconfig::Setting::TypeInt, 4);
    }

    if (!IsExist("granularity")) {
        setSetting("granularity", libconfig::Setting::TypeInt, 16);
    }

    if (!IsExist("clipping")) {
        setSetting("clipping", libconfig::Setting::TypeInt, 0);
    }

    if (!IsExist("vsync")) {
        setSetting("vsync", libconfig::Setting::TypeBoolean, true);
    }

    if (!IsExist("grid")) {
        setSetting("grid", libconfig::Setting::TypeBoolean, false);
    }

    if (!IsExist("debug")) {
        setSetting("debug", libconfig::Setting::TypeInt, 0);
    }

    if (!IsExist("execution_time")) {
        setSetting("execution_time", libconfig::Setting::TypeInt, 0);
    }

    if (!IsExist("workers_count")) {
        setSetting("workers_count", libconfig::Setting::TypeInt, 0);
    }
}

void GlobalConfig::Display() const
{
    auto count_workers = getSetting<int>("workers_count") ? getSetting<int>("workers_count") : tbb::task_scheduler_init::default_num_threads();

    std::cout << "Parallell mode (1 -> serial, 2 -> tbb, 3 -> opencl): " << getSetting<int>("parallell") << std::endl;
    std::cout << "Window width: " << getSetting<float>("width") << std::endl;
    std::cout << "Window height: " << getSetting<float>("height") << std::endl;
    std::cout << "Objects count: " << getSetting<int>("objects_count") << std::endl;
    std::cout << "Load objects frequency in time: " << getSetting<float>("load_objects_freq") << std::endl;
    std::cout << "Rooms count: " << getSetting<int>("rooms_count") << std::endl;
    std::cout << "Collision granularity: " << getSetting<int>("granularity") << std::endl;
    std::cout << "Clipping (0 -> no clipping, 1 -> high clipping, 2 -> low clipping): " << getSetting<int>("clipping") << std::endl;
    std::cout << "Workers count: " << count_workers << std::endl;
    std::cout << "Execution Time (0 -> no limit): " << getSetting<int>("execution_time") << std::endl;
    std::cout << "Vsync (limit framerate to monitor): " << getSetting<bool>("vsync") << std::endl;
    std::cout << "Grid mode (not fill polygons): " << getSetting<bool>("grid") << std::endl;
    std::cout << "Debug mode (0 -> no debug, 1 -> test debug, 2 -> performance debug, 3 -> collision debug, 4 -> all debug): " << getSetting<int>("debug") << std::endl;
}

void GlobalConfig::ManageProgramParameters(int argc, char* argv[])
{
    using nextfloor::physics::CollisionEngine;

    auto is_display_config = false;
    auto cnt = 1;
    while (cnt < argc) {
        /* Cant use switch with string */
        const std::string arg(argv[cnt++]);

        /* Clipping */
        if (arg == "-c") {
            const std::string arg2(argv[cnt++]);
            setSetting("clipping", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Debug mode */
        if (arg == "-d") {
            const std::string arg2(argv[cnt++]);
            setSetting("debug", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Execution Time */
        if (arg == "-e") {
            const std::string arg2(argv[cnt++]);
            setSetting("execution_time", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* granularity */
        if (arg == "-g") {
            const std::string arg2(argv[cnt++]);
            setSetting("granularity", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Help */
        if (arg == "-h") {
            std::cout << argv[0] << " can be used with following options who overrides config file" << std::endl;
            std::cout << "-c n   Clipping, 0: no clipping, 1: high clipping, 2: low clipping" << std::endl;
            std::cout << "-d n   Debug mode, 0: no debug, 1: test debug, 2: performance debug, 3: collision debug, 4: all debug" << std::endl;
            std::cout << "-e n   Execution Time, 0: no limit" << std::endl;
            std::cout << "-g n   Granularity on collision computes" << std::endl;
            std::cout << "-h     Display help" << std::endl;
            std::cout << "-l     Display config" << std::endl;
            std::cout << "-o n   Count of objects in rooms" << std::endl;
            std::cout << "-p serial|tbb|opencl" << std::endl
            << "       serial: no parallellism" << std::endl
            << "       tbb: uses intel tbb library" << std::endl
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
            setSetting("objects_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* parallell type */
        if (arg == "-p") {
            const std::string arg2(argv[cnt++]);
            if (arg2 == "serial") {
                setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_SERIAL);
            }

            if (arg2 == "tbb") {
                setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_TBB);
            }

            if (arg2 == "opencl") {
                setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_CL);
            }
        }

        /* Count of rooms */
        if (arg == "-r") {
            const std::string arg2(argv[cnt++]);
            setSetting("rooms_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }

        /* Sequentially load frequency */
        if (arg == "-s") {
            const std::string arg2(argv[cnt++]);
            setSetting("load_objects_freq", libconfig::Setting::TypeFloat, std::stof(arg2));
        }

        /* Vsync */
        if (arg == "-v") {
            const std::string arg2(argv[cnt++]);
            setSetting("vsync", libconfig::Setting::TypeBoolean, std::stoi(arg2) == 1);
        }

        /* Workers count */
        if (arg == "-w") {
            const std::string arg2(argv[cnt++]);
            setSetting("workers_count", libconfig::Setting::TypeInt, std::stoi(arg2));
        }
    }

    if (getSetting<int>("parallell") == CollisionEngine::kPARALLELL_SERIAL) {
        setSetting("workers_count", libconfig::Setting::TypeInt, 1);
    }

    /* Display config values from file */
    if (is_display_config) {
        Display();
        exit(0);
    }
}

GlobalConfig::~GlobalConfig()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} //namespace core

} //namespace nextfloor
