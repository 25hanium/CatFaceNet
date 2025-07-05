from setuptools import setup, find_packages

setup(
    name='CatFaceNet',
    version='0.1.0',
    description='A library to detect cat faces.',
    author='SERAFI',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/CatFaceNet',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
    ],
    package_data={
        'CatFaceNet': ['profile.npy', 'weight.pt'],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
