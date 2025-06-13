cmake_minimum_required(VERSION 3.5)
project(fastsam_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  geometry_msgs
  vision_msgs
  catkin_virtualenv
)

find_package(Wget REQUIRED)
add_custom_command(
  OUTPUT create_models_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_SOURCE_DIR}/models"
  )
add_custom_target(FastSAM-s.pt ALL 
  DEPENDS create_models_dir
  COMMAND ${WGET_EXECUTABLE} -q "https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-s.pt" -O "${PROJECT_SOURCE_DIR}/models/FastSAM-s.pt"
  )
catkin_generate_virtualenv(
  PYTHON_INTERPRETER python3
  INPUT_REQUIREMENTS requirements.txt
  )

catkin_package()

catkin_install_python(PROGRAMS
  scripts/fastsam_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}#/scripts
  VIRTUALENV
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}#/launch
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}#/models
)
