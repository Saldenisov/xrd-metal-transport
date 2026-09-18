#----------------------------------------------------------------------------
# Setup the project

cmake_minimum_required(VERSION 3.16...3.27)
project(saxs)

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable

option(WITH_GEANT4_UIVIS "Build SAXS with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project

include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries

add_executable(saxs saxs.cc ${sources} ${headers})
target_link_libraries(saxs ${Geant4_LIBRARIES})

#----------------------------------------------------------------------------
# Copy all scripts to the build directory, i.e. the directory in which we
# build SAXS. This is so that we can run the executable directly because it
# relies on these scripts being in the current working directory.

set(SAXS_SCRIPTS
  	init_vis.mac
  	vis.mac
	gui.mac
	geom.mac
	myFF.dat
	saxs.in
	saxs_slits.in
	test.in
	scattAnalysis.C
	ADXRD.C
	keele_builtin_fat.mac
	keele_builtin_water.mac
	keele_builtin_collagen.mac
	keele_builtin_mix.mac
	water_20kev_detector.mac
	prepare_keele_ff.py
	plot_form_factors.py
	analyze_root.py
	plot_detector_image.py
)

foreach(_script ${SAXS_SCRIPTS})
	configure_file(
    	${PROJECT_SOURCE_DIR}/${_script}
    	${PROJECT_BINARY_DIR}/${_script}
    	COPYONLY
	)
endforeach()

file(GLOB KEELE_GENERATED_MACROS ${PROJECT_SOURCE_DIR}/keele_*.mac)
foreach(_macro ${KEELE_GENERATED_MACROS})
  get_filename_component(_macro_name ${_macro} NAME)
  configure_file(${_macro} ${PROJECT_BINARY_DIR}/${_macro_name} COPYONLY)
endforeach()

# Keele form-factor files are generated before configuring this example.
file(COPY ${PROJECT_SOURCE_DIR}/data DESTINATION ${PROJECT_BINARY_DIR})
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/results)

#----------------------------------------------------------------------------
# For internal Geant4 use - but has no effect if you build this
# example standalone
# add_custom_target(saxs_custom DEPENDS saxs)

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX

install(TARGETS saxs DESTINATION bin)
