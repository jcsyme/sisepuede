# USE MULTI-STAGE BUILD TO GENERATE SISEPUEDE DOCKER IMAGE 
# - https://pythonspeed.com/articles/conda-docker-image-size/
# 1. build conda and environment 
# 2. use `conda-pack` to transfer to Julia image
# - file can be improved with some work
#   - e.g., move to Julia Alpine, but will require more precise installation


##  (1) BUILD CONDA ENVIRONMENT

FROM continuumio/miniconda3 AS build

# see https://jcristharif.com/conda-docker-tips.html for use of this environment variable
ENV PYTHONDONTWRITEBYTECODE=true

# get and create environment from yaml, then drop unnecessary files 
COPY ./environment_py.yaml environment_py.yaml
RUN conda env create -f environment_py.yaml \
    #&& find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    #&& find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && conda clean -afy

# get conda-pack
RUN conda install -c conda-forge conda-pack

# next, use conda-pack to build virtual environment at /venv 
RUN conda-pack -n sisepuede -o /tmp/env.tar \
    && mkdir /venv \
    && cd /venv \
    && tar -xf /tmp/env.tar \
    && rm /tmp/env.tar

# unpack it into bin
RUN /venv/bin/conda-unpack


##  (2) COPY TO JULIA BUILD

FROM julia:1.8.5-bullseye as final
COPY --from=build /venv /venv
WORKDIR /sisepuede

# COPY JULIA DIRECTORY OVER FIRST
# - stable code
# - doesn't have to be rebuilt often
# - necessary for setting up julia environment
COPY ./julia ./julia

# UPDATE AND GET KEY TOOLS
RUN apt-get update \
    && apt-get install -y curl git g++ gcc build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# SETUP JULIA
RUN julia -e 'using Pkg; \
    cd("julia"); \
    Pkg.activate(".");\
    Pkg.rm("NemoMod"); \
    Pkg.rm("Gurobi"); \
    Pkg.rm("GAMS"); \
    Pkg.add(url = "https://github.com/sei-international/NemoMod.jl"); \
    Pkg.instantiate()'

# ADD JULIA TO PYTHON (MAY SHIFT TO INSTALL PYCALL ABOVE)
SHELL ["/bin/bash", "-c"]
RUN source /venv/bin/activate \
    && pip install julia \
    && python -c "import julia; julia.install()"

# COPY REST OF SISEPUEDE OVER
RUN mkdir ./python
COPY ./docs ./docs
COPY ./python/*.py ./python/
COPY ./ref ./ref
COPY ./sisepuede.config ./sisepuede.config
RUN mkdir ./out \
    && chmod 777 ./out


# SETUP CONDA IN BASH AND SET ENTRYPOINT
# - feed a script from host using -c
SHELL ["conda", "run", "-n", "sisepuede", "/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
