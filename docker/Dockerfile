FROM nvcr.io/nvidia/pytorch:19.06-py3
WORKDIR /workspace/EagleEye
COPY requirements_v3.txt requirements.txt
RUN pip install pip -U
RUN pip install --upgrade setuptools
RUN pip install -U --ignore-installed wrapt==1.11.1 enum34 simplejson netaddr pyyaml msgpack==0.5.6
RUN pip install -r requirements.txt
