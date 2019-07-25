/**
 *  @file config_file.cc
 *  @brief ConfigFile Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/config_file.h"

#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <tbb/tbb.h>
#include <cassert>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

static std::string GetConfigFilePath()
{
    struct stat buffer;
    return stat("config/nextfloor.ini", &buffer) == 0 ? "config/nextfloor.ini" : "config/nextfloor.ini.default";
}

static void HandleParseConfigFileIOError(std::string config_file, const libconfig::FileIOException& file_io_exception)
{
    std::cerr << "I/O error while reading file, config parser cancelled: " << config_file << std::endl;
    std::cerr << "Exception: " << file_io_exception.what() << std::endl;
    CommonServices::getExit()->ExitOnError();
}

static void HandleParseConfigFileParsingError(const libconfig::ParseException& parse_exception)
{
    std::cerr << "Parse error at " << parse_exception.getFile() << ":" << parse_exception.getLine()
        << " - " << parse_exception.getError() << std::endl;
    CommonServices::getExit()->ExitOnError();
}

}

ConfigFile::ConfigFile()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void ConfigFile::Initialize()
{
    ParseConfigFile();
    InitDefaultValues();
}

void ConfigFile::ParseConfigFile()
{
    std::string config_file = GetConfigFilePath();

    try {
        config.readFile(config_file.c_str());
    }
    catch(const libconfig::FileIOException& file_io_exception) {
        HandleParseConfigFileIOError(config_file, file_io_exception);
    }
    catch(const libconfig::ParseException& parse_exception) {
        HandleParseConfigFileParsingError(parse_exception);
    }
}

void ConfigFile::InitDefaultValues()
{
    SetDefaultParallellValueIfEmpty();
    SetDefaultParallellThreadCountValueIfEmpty();
    SetDefaultWidthValueIfEmpty();
    SetDefaultHeightValueIfEmpty();
    SetDefaultObjectCountValueIfEmpty();
    SetDefaultLoadObjectsFreqValueIfEmpty();
    SetDefaultRoomCountValueIfEmpty();
    SetDefaultCollisionGranularityValueIfEmpty();
    SetDefaultClippingValueIfEmpty();
    SetDefaultVsyncValueIfEmpty();
    SetDefaultGridModeValueIfEmpty();
    SetDefaultDebugVerbosityValueIfEmpty();
    SetDefaultExecutionTimeValueIfEmpty();
    SetDefaultUniverseFactoryValueIfEmpty();
}

void ConfigFile::SetDefaultParallellValueIfEmpty()
{
    // using nextfloor::physics::CollisionEngine;

    // if (!IsExist("parallell")) {
    //     setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_SERIAL);
    // }
}

void ConfigFile::SetDefaultParallellThreadCountValueIfEmpty()
{
    if (!IsExist("workers_count")) {
        setSetting("workers_count", libconfig::Setting::TypeInt, 0);
    }
}

void ConfigFile::SetDefaultWidthValueIfEmpty()
{
    if (!IsExist("width")) {
        setSetting("width", libconfig::Setting::TypeFloat, 800.0f);
    }
}

void ConfigFile::SetDefaultHeightValueIfEmpty()
{
    if (!IsExist("height")) {
        setSetting("height", libconfig::Setting::TypeFloat, 600.0f);
    }
}

void ConfigFile::SetDefaultObjectCountValueIfEmpty()
{
    if (!IsExist("objects_count")) {
        setSetting("objects_count", libconfig::Setting::TypeInt, 16);
    }
}

void ConfigFile::SetDefaultLoadObjectsFreqValueIfEmpty()
{
    if (!IsExist("load_objects_freq")) {
        // 0 => loads all objects at start
        setSetting("load_objects_freq", libconfig::Setting::TypeFloat, 0.0f);
    }
}

void ConfigFile::SetDefaultRoomCountValueIfEmpty()
{
    if (!IsExist("rooms_count")) {
        setSetting("rooms_count", libconfig::Setting::TypeInt, 4);
    }
}

void ConfigFile::SetDefaultCollisionGranularityValueIfEmpty()
{
    if (!IsExist("granularity")) {
        setSetting("granularity", libconfig::Setting::TypeInt, 16);
    }
}

void ConfigFile::SetDefaultClippingValueIfEmpty()
{
    if (!IsExist("clipping")) {
        setSetting("clipping", libconfig::Setting::TypeInt, 0);
    }
}

void ConfigFile::SetDefaultVsyncValueIfEmpty()
{
    if (!IsExist("vsync")) {
        setSetting("vsync", libconfig::Setting::TypeBoolean, true);
    }
}

void ConfigFile::SetDefaultGridModeValueIfEmpty()
{
    if (!IsExist("grid")) {
        setSetting("grid", libconfig::Setting::TypeBoolean, false);
    }
}

void ConfigFile::SetDefaultDebugVerbosityValueIfEmpty()
{
    if (!IsExist("debug")) {
        setSetting("debug", libconfig::Setting::TypeInt, 0);
    }
}

void ConfigFile::SetDefaultExecutionTimeValueIfEmpty()
{
    if (!IsExist("execution_time")) {
        setSetting("execution_time", libconfig::Setting::TypeInt, 0);
    }
}

void ConfigFile::SetDefaultUniverseFactoryValueIfEmpty()
{
    // using nextfloor::factory::UniverseFactory;

    // if (!IsExist("factory_type")) {
    //     setSetting("factory_type", libconfig::Setting::TypeInt, UniverseFactory::kUNIVERSEFACTORY_DEMO);
    // }
}


void ConfigFile::Display() const
{
    auto count_workers = GetWorkersCount();

    std::cout << "Parallell mode (1 -> serial, 2 -> tbb, 3 -> opencl): " << getSetting<int>("parallell") << std::endl;
    std::cout << "Window width: " << getSetting<float>("width") << std::endl;
    std::cout << "Window height: " << getSetting<float>("height") << std::endl;
    std::cout << "Objects count: " << getSetting<int>("objects_count") << std::endl;
    std::cout << "Load objects frequency in time: " << getSetting<float>("load_objects_freq") << std::endl;
    std::cout << "Rooms count: " << getSetting<int>("rooms_count") << std::endl;
    std::cout << "NearerCollisionEngine granularity: " << getSetting<int>("granularity") << std::endl;
    std::cout << "Clipping (0 -> no clipping, 1 -> high clipping, 2 -> low clipping): " << getSetting<int>("clipping") << std::endl;
    std::cout << "Workers count: " << count_workers << std::endl;
    std::cout << "Execution Time (0 -> no limit): " << getSetting<int>("execution_time") << std::endl;
    std::cout << "Vsync (limit framerate to monitor): " << getSetting<bool>("vsync") << std::endl;
    std::cout << "PlacesGrid mode (not fill polygons): " << getSetting<bool>("grid") << std::endl;
    std::cout << "Program mode : " << getSetting<int>("factory_type") << std::endl;
    std::cout << "Debug mode (0 -> no debug, 1 -> test debug, 2 -> performance debug, 3 -> collision debug, 4 -> all debug): " << getSetting<int>("debug") << std::endl;
}

int ConfigFile::GetWorkersCount() const
{
    return getSetting<int>("workers_count") ? getSetting<int>("workers_count") : tbb::task_scheduler_init::default_num_threads();
}

void ConfigFile::ManageProgramParameters(int argc, char* argv[])
{
    const std::string program_name(argv[0]);
    auto is_display_config = false;
    auto cnt = 1;

    while (cnt < argc) {

        const std::string parameter_name(argv[cnt++]);

        if (IsHelpParameter(parameter_name)) {
            DisplayHelp(program_name);
            CommonServices::getExit()->ExitOnSuccess();
        }

        if (IsDisplayConfigParameter(parameter_name)) {
            is_display_config = true;
            continue;
        }

        /* name/value parameters need one more word */
        assert(cnt < argc);
        const std::string parameter_value(argv[cnt++]);

        ManageClippingParameter(parameter_name, parameter_value);
        ManageDebugParameter(parameter_name, parameter_value);
        ManageExecutionTimeParameter(parameter_name, parameter_value);
        ManageGranularityParameter(parameter_name, parameter_value);
        ManageObjectCountParameter(parameter_name, parameter_value);
        ManagePrallellAlgoTypeParameter(parameter_name, parameter_value);
        ManageRoomCountParameter(parameter_name, parameter_value);
        ManageLoadObjectFrequencyParameter(parameter_name, parameter_value);
        ManageVsyncParameter(parameter_name, parameter_value);
        ManageWorkerCountParameter(parameter_name, parameter_value);
        ManageUniverseFactoryTypeParameter(parameter_name, parameter_value);
    }

    EnsureCoherentWorkerSetting();

    if (is_display_config) {
        Display();
        CommonServices::getExit()->ExitOnSuccess();
    }
}

bool ConfigFile::IsDisplayConfigParameter(const std::string& parameter_name) const
{
    if (parameter_name == "-l") {
        return true;
    }
    return false;
}

bool ConfigFile::IsHelpParameter(const std::string& parameter_name) const
{
    if (parameter_name == "-h") {
        return true;
    }
    return false;
}

void ConfigFile::DisplayHelp(const std::string& command_name) const
{
    std::cout << command_name << " can be used with following options who overrides config file" << std::endl;
    std::cout << "-c n   Clipping, 0: no clipping, 1: high clipping, 2: low clipping" << std::endl;
    std::cout << "-d n   Debug mode, 0: no debug, 1: test debug, 2: performance debug, 3: collision debug, 4: all debug" << std::endl;
    std::cout << "-f demo|random program mode" << std::endl;
    std::cout << "-e n   Execution Time, 0: no limit" << std::endl;
    std::cout << "-g n   Granularity on collision computes" << std::endl;
    std::cout << "-h     Display help" << std::endl;
    std::cout << "-l 1|0 Enable/Disable display config" << std::endl;
    std::cout << "-o n   Count of objects in rooms" << std::endl;
    std::cout << "-p serial|tbb|opencl" << std::endl
    << "       serial: no parallellism" << std::endl
    << "       tbb: uses intel tbb library" << std::endl
    << "       opencl: uses opencl for collision computes" << std::endl;
    std::cout << "-r n   Count of rooms" << std::endl;
    std::cout << "-s n.m Load objects frequency, 0: generates all objects at start" << std::endl;
    std::cout << "-v 1|0 Enable/Disable vsync" << std::endl;
    std::cout << "-w n   Workers (cpu core) count (disabled if -p serial), 0: no limit, all cpu cores" << std::endl;
}

void ConfigFile::ManageClippingParameter(const std::string& parameter_name,
                                           const std::string& parameter_value)
{
    if (parameter_name == "-c") {
        setSetting("clipping", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void ConfigFile::ManageDebugParameter(const std::string& parameter_name,
                                        const std::string& parameter_value)
{
    if (parameter_name == "-d") {
        setSetting("debug", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void ConfigFile::ManageExecutionTimeParameter(const std::string& parameter_name,
                                        const std::string& parameter_value)
{
    if (parameter_name == "-e") {
        setSetting("execution_time", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void ConfigFile::ManageGranularityParameter(const std::string& parameter_name,
                                              const std::string& parameter_value)
{
    if (parameter_name == "-g") {
        setSetting("granularity", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void ConfigFile::ManageObjectCountParameter(const std::string& parameter_name,
                                              const std::string& parameter_value)
{
    if (parameter_name == "-o") {
        setSetting("objects_count", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void ConfigFile::ManagePrallellAlgoTypeParameter(const std::string& parameter_name,
                                                  const std::string& parameter_value)
{
    // using nextfloor::physics::CollisionEngine;

    // if (parameter_name == "-p") {
    //     if (parameter_value == "serial") {
    //         setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_SERIAL);
    //     }

    //     if (parameter_value == "tbb") {
    //         setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_TBB);
    //     }

    //     if (parameter_value == "opencl") {
    //         setSetting("parallell", libconfig::Setting::TypeInt, CollisionEngine::kPARALLELL_CL);
    //     }
    // }
}

void ConfigFile::ManageRoomCountParameter(const std::string& parameter_name,
                                            const std::string& parameter_value)
{
    if (parameter_name == "-r") {
        setSetting("rooms_count", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void ConfigFile::ManageLoadObjectFrequencyParameter(const std::string& parameter_name,
                                                      const std::string& parameter_value)
{
    if (parameter_name == "-s") {
        setSetting("load_objects_freq", libconfig::Setting::TypeFloat, std::stof(parameter_value));
    }
}

void ConfigFile::ManageVsyncParameter(const std::string& parameter_name,
                                        const std::string& parameter_value)
{
    if (parameter_name == "-v") {
        setSetting("vsync", libconfig::Setting::TypeBoolean, std::stoi(parameter_value) == 1);
    }
}

void ConfigFile::ManageWorkerCountParameter(const std::string& parameter_name,
                                              const std::string& parameter_value)
{
    if (parameter_name == "-w") {
        setSetting("workers_count", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void ConfigFile::EnsureCoherentWorkerSetting()
{
 //   using nextfloor::physics::CollisionEngine;

    /* Disable tbb usage when serial parallel algorithm is setted */
   // if (getSetting<int>("parallell") == CollisionEngine::kPARALLELL_SERIAL) {
        setSetting("workers_count", libconfig::Setting::TypeInt, 1);
    //}
}

void ConfigFile::ManageUniverseFactoryTypeParameter(const std::string& parameter_name,
                                                    const std::string& parameter_value)
{
    // using nextfloor::factory::UniverseFactory;

    // if (parameter_name == "-f") {
    //     if (parameter_value == "demo") {
    //         setSetting("factory_type", libconfig::Setting::TypeInt, UniverseFactory::kUNIVERSEFACTORY_DEMO);
    //     }

    //     if (parameter_value == "random") {
    //         setSetting("factory_type", libconfig::Setting::TypeInt, UniverseFactory::kUNIVERSEFACTORY_RANDOM);
    //     }
    // }
}

ConfigFile::~ConfigFile()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} //namespace core

} //namespace nextfloor
