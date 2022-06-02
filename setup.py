from setuptools import setup,find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()


# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='orsvm', 
    version='1.0.4',
    description='SVM with Orthogonal Kernel functions of fractional order and normal',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AmirHoseinHadian/orsvm',
    author='Amir Hossein Hadian Rasenan, Sherwin Nedaei Janbesaraei, Amirreza Azmoon, Mohammad Akhavan',
    author_email='amir.h.hadian@gmail.com',
    classifiers=[
	'Development Status :: 4 - Beta',	
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='SVM, Orthogonal Polynomials, Classification, Chebyshev, Legendre, Gegenbauer, Jacobi',
    packages=['orsvm'],
    install_requires=[
            'Cython',
            'cvxopt',
            'numpy',
            'pandas',
            'scikit_learn',
	    'sphinx',
	    'ipykernel',
	    'nbsphinx'
	    
        ],
    python_requires='>=3.8',
)
