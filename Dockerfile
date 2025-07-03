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
COPY ./environment_docker.yaml environment_docker.yaml
RUN conda env create -f environment_docker.yaml \
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

FROM julia:1.11.5-bullseye AS final
COPY --from=build /venv /venv
WORKDIR /sisepuede

# UPDATE AND GET KEY TOOLS
RUN apt-get update \
    && apt-get install -y curl git g++ gcc vim build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# ADD JULIA TO PYTHON
# import after installation to ensure julia is installed
SHELL ["/bin/bash", "-c"]
RUN source /venv/bin/activate \
    && pip install juliacall==0.9.25 \
    && pip install juliapkg==0.1.17 \
    && pip install git+https://github.com/jcsyme/sisepuede/

RUN source /venv/bin/activate \
    && python -c "import sisepuede.manager.sisepuede_file_structure as sfs; \
    import sisepuede.manager.sisepuede_models as sm; \
    file_struct = sfs.SISEPUEDEFileStructure(); \
    models_all = sm.SISEPUEDEModels(file_struct.model_attributes, allow_electricity_run = True, fp_julia = file_struct.dir_jl, fp_nemomod_reference_files = file_struct.dir_ref_nemo, fp_nemomod_temp_sqlite_db = file_struct.fp_sqlite_tmp_nemomod_intermediate, );";


# SETUP CONDA IN BASH AND SET ENTRYPOINT
# - feed a script from host using -c
SHELL ["conda", "run", "-n", "sisepuede", "/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]




