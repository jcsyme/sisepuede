#
#  THINGS TO DO ON SETUP
#

#
# set juliapkg path environment variable here to point to julia json
# install julia 
#

import os, os.path
from setuptools import setup




# read in requirements and install
dir_cur = os.path.dirname(os.path.realpath(__file__))
fp_julia = os.path.join(dir_cur, "julia")
fp_requirements = os.path.join(dir_cur, "requirements.txt")

flag_version = "python_version"

if os.path.isfile(fp_requirements):
    with open(fp_requirements, "r") as fl:
        reqs = fl.readlines()
        
        # get python version
        #py_version = [x.replace(flag_version, "") for x in reqs if flag_version in x]
        #py_version = py_version[0] if (len(py_version) > 0) else None
        reqs = [x for x in reqs if flag_version not in x]


# call setup
setup(
    author = "James Syme",
    author_email = "jsyme@tec.mx",
    description = "SImulation of SEctoral Pathways and Uncertainty Exploration for DEcarbonization is a multi-sector, integrated emission accounting and modeling framework for evalauting decarboniation policies under uncertainty.",
    include_package_data = True,
    license = "MIT",
    name = "SISEPUEDE",
    packages = [
        "sisepuede",
        "sisepuede.cloud",
        "sisepuede.command_line",
        "sisepuede.core",
        "sisepuede.data_management",
        "sisepuede.geo",
        "sisepuede.manager",
        "sisepuede.models",
        "sisepuede.pipeline",
        "sisepuede.plotting",
        "sisepuede.transformers",
        "sisepuede.transformers.lib",
        "sisepuede.utilities",
        "sisepuede.utilities.data_support"
    ],
    package_data = {
        "": [
            "attributes/**",
            "docs/**",
            "julia/**",
            "ref/**",
            "sisepuede.config"
        ]
    },
    #python_requires = py_version,
    url = "http://github.com/jcsyme/sisepuede",
    version = "1.3.2",
    zip_safe = False
)



