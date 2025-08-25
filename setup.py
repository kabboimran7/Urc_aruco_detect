from setuptools import find_packages, setup

package_name = 'urc_aruco_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@todo.com',
    description='Aruco detection and calibration package for URC',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'camera_publisher = urc_aruco_detector.camera_publisher_node:main',
            'calibration_service = urc_aruco_detector.calibration_service_node:main',
            'aruco_detector = urc_aruco_detector.aruco_detector_node:main',
        ],
    },
)