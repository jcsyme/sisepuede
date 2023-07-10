# NOTE: THIS APPROACH WORKS—NEED TO TRY SOME ADDITIONAL PIECES, LIKE CACHING PRE-BUILD AND MOVING TO JULIA ALPINE

# MULTI-STAGE: build conda and environment first, then use conda-pack to transfer to Julia image
# https://pythonspeed.com/articles/conda-docker-image-size/
FROM continuumio/miniconda3 AS build

# see https://jcristharif.com/conda-docker-tips.html for decision
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



FROM julia:1.8.5-bullseye as final
COPY --from=build /venv /venv
WORKDIR /sisepuede

# COPY SISEPUEDE COMPONENTS OVER
COPY ./docs ./docs
COPY ./python ./python
COPY ./julia ./julia
COPY ./ref ./ref
COPY ./sisepuede.config ./sisepuede.config
RUN mkdir ./out \
    && chmod 777 ./out

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

# TEMPORARY CMD
#COPY sisepuede_exec.sh .
SHELL ["conda", "run", "-n", "sisepuede", "/bin/bash", "-c"]
#WORKDIR /sisepuede/python
#ENV JULIA_NUM_THREADS=`$(nproc)\*2`
#ENV LD_PRELOAD=/usr/local/julia/lib/julia/libstdc++.so.6
ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["/bin/bash", "-c", "source ./sisepuede_exec.sh"]
#CMD ["-c", "chmod +x /venv/bin/activate", "-c", "/venv/bin/activate", "python", "sisepuede_cl.py"]