#!/bin/bash
#
# Launch performance tests with some metrics in output 
# @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)

NCORES=0
REPORTFOLDER=tmp
NOW="$(date +%y%m%d%H%I)"
OPENCL=1
NBOBJ=32
NBR=4
GRAN=64
TESTS=2

# Ensure we are into the good location
if [[ ! -f scripts/${0##.*/} ]]; then
    echo "Bad folder, you must be into project root directory"
    echo "And launch ./scripts/./${0##.*/}"
    exit 1
fi

# display help
testperfhelp() {
    cat <<EOF
Usage: $0 [options]

-h          Some help text.
-e path     Report folder (default $PWD/tmp/)
-g n        Set granularity for collision computes (default is 64)
-nocl       Disable tests using opencl for parallell collision computes
-o n        Set count of moving objects into rooms (default is 32)
-r n        Set count of rooms (default is 4)
-t n        Select test(s) to execute (0 => no seq load and during 60s, 1 => seq load, 2 => both tests)
-w n        Set workers (cpu core) number (default is 0)
EOF
exit 0
}

# Parse cmd options
while (($# > 0)); do
  case "$1" in
    -h)
      shift
      testperfhelp
      ;;
    -e)
      shift
      REPORTFOLDER=$1
      shift
      ;;
    -g)
      shift
      GRAN=$1
      shift
      ;;
    -nocl)
      shift
      OPENCL=0
      ;;
    -o)
      shift
      NBOBJ=$1
      shift
      ;;
    -r)
      shift
      NBR=$1
      shift
      ;;
    -t)
      shift
      TESTS=$1
      shift
      ;;
    -w)
      shift
      NCORES="$1"
      shift
      ;;
    *)
      testperfhelp
      shift
      ;;
  esac
done

# load library path
if [[ -f /usr/bin/sw_vers ]]; then
    source scripts/cilk_osx_vars.sh
else
    source scripts/cilk_linux_vars.sh
fi

# prepare report folder
mkdir -p $REPORTFOLDER
rm -rf ${REPORTFOLDER}/test_${NOW}
mkdir -p ${REPORTFOLDER}/test_${NOW}


###### PART 1: SERIAL TEST #######

# perf debug
# no parallellism
# NBOBJ (32) objects by room, NBR (4) rooms (NBR*NBOBJ Moving Objects in all)
# GRAN (64) computes for collision
# no clipping (all objects are displayed)
# 60 seconds execution
# no vsync
# no sequentially load (all objects are displayed at start)
# NCORES cpu cores
if ((TESTS % 2 == 0)); then
./bin/./engine -d 1 -p serial -o $NBOBJ -r $NBR -g $GRAN -c 0 -e 60 -v 0 -s 0 -l > ${REPORTFOLDER}/test_${NOW}/test0_1_settings
./bin/./engine -d 1 -p serial -o $NBOBJ -r $NBR -g $GRAN -c 0 -e 60 -v 0 -s 0 > ${REPORTFOLDER}/test_${NOW}/test0_1_report
fi

# perf debug
# no parallellism
# Until 2*NBOBJ (64) objects by room, NBR (4) rooms
# GRAN (64) computes for collision
# no clipping (all objects are displayed)
# 2*NBOBJ*10 seconds execution
# no vsync
# Sequentially load: one new object by room each 10s
if ((TESTS >= 1)); then
./bin/./engine -d 1 -p serial -o $((2*NBOBJ)) -r $NBR -g $GRAN -c 0 -e $((10*2*NBOBJ)) -v 0 -s 10 -l > ${REPORTFOLDER}/test_${NOW}/test1_1_settings
./bin/./engine -d 1 -p serial -o $((2*NBOBJ)) -r $NBR -g $GRAN -c 0 -e $((10*2*NBOBJ)) -v 0 -s 10 > ${REPORTFOLDER}/test_${NOW}/test1_1_report
fi

###### PART 2: OPENCL TEST #######

if ((OPENCL == 1)); then
# perf debug
# opencl parallellism for collision computes
# NBOBJ (32) objects by room, NBR (4) rooms (NBR*NBOBJ Moving Objects in all)
# GRAN (64) computes for collision
# no clipping (all objects are displayed)
# 60 seconds execution
# no vsync
# no sequentially load (all objects are displayed at start)
# NCORES cpu cores
if ((TESTS % 2 == 0)); then
./bin/./engine -d 1 -p opencl -o $NBOBJ -r $NBR -g $GRAN -c 0 -e 60 -v 0 -s 0 -l > ${REPORTFOLDER}/test_${NOW}/test0cl_settings
./bin/./engine -d 1 -p opencl -o $NBOBJ -r $NBR -g $GRAN -c 0 -e 60 -v 0 -s 0 > ${REPORTFOLDER}/test_${NOW}/test0cl_report
fi

# perf debug
# opencl parallellism
# Until 2*NBOBJ (64) objects by room, NBR (4) rooms
# GRAN (64) computes for collision
# no clipping (all objects are displayed)
# 2*NBOBJ*10 seconds execution
# no vsync
# Sequentially load: one new object by room each 10s
if ((TESTS >= 1)); then
./bin/./engine -d 1 -p opencl -o $((2*NBOBJ)) -r $NBR -g $GRAN -c 0 -e $((10*2*NBOBJ)) -v 0 -s 10 -l > ${REPORTFOLDER}/test_${NOW}/test1cl_settings
./bin/./engine -d 1 -p opencl -o $((2*NBOBJ)) -r $NBR -g $GRAN -c 0 -e $((10*2*NBOBJ)) -v 0 -s 10 > ${REPORTFOLDER}/test_${NOW}/test1cl_report
fi

fi


###### PART 3: CILKPLUS TEST #######

NC=2
((NCORES == 0)) && NC=0
while ((NC <= NCORES)); do
# perf debug
# cilkplus parallellism
# NBOBJ (32) objects by room, NBR (4) rooms (NBR*NBOBJ Moving Objects in all)
# GRAN (64) computes for collision
# no clipping (all objects are displayed)
# 60 seconds execution
# no vsync
# no sequentially load (all objects are displayed at start)
# NCORES cpu cores
if ((TESTS % 2 == 0)); then
./bin/./engine -d 1 -p cilkplus -o $NBOBJ -r $NBR -g $GRAN -c 0 -e 60 -v 0 -s 0 -w $NC -l > ${REPORTFOLDER}/test_${NOW}/test0_${NC}_settings
./bin/./engine -d 1 -p cilkplus -o $NBOBJ -r $NBR -g $GRAN -c 0 -e 60 -v 0 -s 0 -w $NC > ${REPORTFOLDER}/test_${NOW}/test0_${NC}_report
fi

# perf debug
# cilkplus parallellism
# Until 2*NBOBJ (64) objects by room, NBR (4) rooms
# GRAN (64) computes for collision
# no clipping (all objects are displayed)
# 2*NBOBJ*10 seconds execution
# no vsync
# Sequentially load: one new object by room each 10s
if ((TESTS >= 1)); then
./bin/./engine -d 1 -p cilkplus -o $((2*NBOBJ)) -r $NBR -g $GRAN -c 0 -e $((10*2*NBOBJ)) -v 0 -s 10 -w $NC -l > ${REPORTFOLDER}/test_${NOW}/test1_${NC}_settings
./bin/./engine -d 1 -p cilkplus -o $((2*NBOBJ)) -r $NBR -g $GRAN -c 0 -e $((10*2*NBOBJ)) -v 0 -s 10 -w $NC > ${REPORTFOLDER}/test_${NOW}/test1_${NC}_report
fi

NC=$((NC+2))
done
