from setuptools import setup,find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="orsvm", # Replace with your own username
    version="0.1.1",
    author="Amir Hossein Hadian Rasenan, Sherwin Nedaei Janbesaraei, Amirreza Azmoon, Mohammad Akhavan",
    author_email="amir.h.hadian@gmail.com",
    description="SVM with Orthogonal Kernel functions of fractional order and normal",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="Empty",
    classifiers=[	
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=['orsvm'],
    install_requires=[
            "Cython",
            "cvxopt",
            "numpy",
            "pandas",
            "scikit_learn"
        ],
    python_requires='>=3.8',
)
