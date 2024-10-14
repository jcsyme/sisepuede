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

if os.path.isfile(fp_requirements):
    with open(fp_requirements, "r") as fl:
        reqs = fl.readlines()
        reqs = [x for x in reqs if "python_version" not in x]

# call setup
setup(
    author = "James Syme",
    author_email = "jsyme@tec.mx",
    description = "SImulation of SEctoral Pathways and Uncertainty Exploration for DEcarbonization is a multi-sector, integrated emission accounting and modeling framework for evalauting decarboniation policies under uncertainty.",
    license = "MIT",
    name = "SISEPUEDE",
    packages = ["sisepuede"],
    url = "http://github.com/jcsyme/sisepuede",
    version = "1.1",
    zip_safe = False
)


#
#   CHECK JULIAPKG AND INSTALL JULIA+JULIA REQS
#
if os.path.exists(fp_julia):
    os.environ["PYTHON_JULIAPKG_PROJECT"] = fp_julia
    import juliapkg

