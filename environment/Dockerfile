# hash:sha256:0be5a1b162d53e9286880f943e20c0bc69604667b87cc64d5f23df6b5bdce385
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=conda-forge \
        jupyter==1.0.0 \
        jupyterlab==3.5.0 \
        matplotlib==3.6.2 \
        matplotlib-base==3.6.2 \
        numpy==1.23.5 \
        numpy-base==1.23.5 \
        pandas==1.5.2 \
        plotly==5.13.1 \
        pyomo==6.8.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    geopandas==0.12.2 \
    tables==3.8.0

RUN pip3 install -U --no-cache-dir \
    ortools==9.5.2237
