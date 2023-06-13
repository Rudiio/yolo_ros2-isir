from setuptools import setup
from setuptools import find_packages

package_name = 'yolov7'
yolov7_subpackage1 = 'includes/models'

setup(
    name=package_name,
    version='0.0.0',
    # packages=[package_name,yolov7_subpackage1],
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rudio',
    maintainer_email='rudio.fida_cyrille@etu.sorbonne-universite.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov7=yolov7.yolov7:ros_main'
        ],
    },
)
