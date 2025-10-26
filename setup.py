from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake required")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        # Use your LLVM/MLIR paths
        LLVM_DIR = "/usr/local/opt/llvm"
        MLIR_DIR = "/usr/local/opt/llvm/lib/cmake/mlir"
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/iam',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DLLVM_DIR={LLVM_DIR}/lib/cmake/llvm',
            f'-DMLIR_DIR={MLIR_DIR}',
        ]
        
        build_args = ['--', '-j4']
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

setup(
    name='iam',
    version='0.1.0',
    packages=['iam'],
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension('iam._iam_core')],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=['torch>=2.0.0', 'numpy>=1.21.0'],
)
