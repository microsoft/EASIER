# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import setup, find_packages
from torch.utils import cpp_extension

entry_points = {
    "console_scripts": [
        # EXE_NAME = PY_MOD:FUNC_NAME
        "easierrun = easier.launcher.main:main"
    ]
}

setup(
    name="easier",
    version="0.0.1",
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        "easier.launcher": ["*.sh"]
    },
    entry_points=entry_points,
    ext_modules=[
        cpp_extension.CppExtension(
            name='easier.cpp_extension',
            sources=[
                'csrc/init.cpp',
                'csrc/triangular_mesh.cpp',
                'csrc/distpart.cpp',
            ],
            extra_compile_args=['-fopenmp']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
