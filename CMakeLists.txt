cmake_minimum_required(VERSION 3.8)
project(urc_aruco_detector)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME} "srv/StartCalibration.srv")

ament_python_install_package()

ament_package()