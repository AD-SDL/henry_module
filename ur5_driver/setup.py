from setuptools import setup
import os
from glob import glob

package_name = 'ur5_driver'

setup(
    name = package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kendrick, Doga Ozgulbas',
    maintainer_email='dozgulbas@anl.gov',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ur5_driver = ur5_driver.ur5_driver:main',
            'urx_packages = ur5_driver.urx_packages.urx:main',

        ],
    },
)