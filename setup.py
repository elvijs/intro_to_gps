"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []

test_requirements = ["pytest", ]

setup(
    author="Elvijs Sarkans",
    author_email='elvijs.sarkans@gmail.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="A gentle introduction to GPs - talk accompaniment",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    name='Intro to GPs',
    packages=['src'],
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/elvijs/intro_to_gps',
    version='0.1.0',
    zip_safe=False,
)
