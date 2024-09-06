from setuptools import find_packages, setup

package_name = 'yolov5_ros2'

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
    maintainer='mayank',
    maintainer_email='sharmamayank1301@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_node = yolov5_ros2.yolov5_ros2_node:main',
            'yolov8_node = yolov5_ros2.yolov8_optimized:main'

        ],
    },
)
