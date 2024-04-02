FROM nvidia/cuda@sha256:0a90df2e70c3359d51a18baf924d3aa65570f02121646daa6749de6d2b46a464

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update

# Python 3.6
RUN apt-get install -y libssl-dev \
    zlib1g-dev \
    libjpeg-dev

RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz \
 && tar xvf Python-3.6.15.tgz \
 && cd Python-3.6.15 \
 && ./configure --prefix=/usr --enable-optimizations --enable-shared \
 && make altinstall

RUN if [ -f /usr/bin/python3 ]; then rm /usr/bin/python3; fi && ln -s /usr/bin/python3.6 /usr/bin/python3
RUN if [ -f /usr/bin/python ]; then rm /usr/bin/python; fi && ln -s /usr/bin/python3.6 /usr/bin/python
RUN if [ -f /usr/bin/pip3 ]; then rm /usr/bin/pip3; fi && ln -s /usr/bin/pip3.6 /usr/bin/pip3
RUN if [ -f /usr/bin/pip ]; then rm /usr/bin/pip; fi && ln -s /usr/bin/pip3.6 /usr/bin/pip

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

# Create Mujoco subdir.
RUN mkdir /root/.mujoco
COPY mjkey.txt /root/.mujoco/mjkey.txt

# Prerequisites
RUN apt-get install \
  libosmesa6-dev \
  libgl1-mesa-glx \
  libglfw3 \
  libglew-dev \
  patchelf \
  gcc \
  unzip -y \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  vim \
  openssh-server \
  openmpi-bin \
  openmpi-common \
  openssh-client \
  libopenmpi-dev

# Download and install mujoco.
RUN wget https://www.roboti.us/download/mujoco200_linux.zip
RUN unzip mujoco200_linux.zip
RUN rm mujoco200_linux.zip
RUN mv mujoco200_linux /root/.mujoco/mujoco200
RUN wget -P /root/.mujoco/mujoco200/bin/ https://roboti.us/file/mjkey.txt

RUN rm /usr/bin/lsb_release
# Add LD_LIBRARY_PATH environment variable.
ENV LD_LIBRARY_PATH "/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}"
RUN echo 'export LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}' >> /etc/bash.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat:${LD_LIBRARY_PATH}' >> /etc/bash.bashrc

# Finally, install mujoco_py.
RUN pip install mujoco_py==2.0.2.8
RUN pip install gym==0.12.5 protobuf==3.7.1 grpcio==1.20.1 imageio==2.5.0 tensorflow-gpu==1.13.1 pyquaternion joblib==0.13.2 opencv-python==4.1.0.25 mpi4py torch

WORKDIR /

RUN git clone https://github.com/mingfeisun/DeepMimic_mujoco.git

SHELL ["/bin/bash", "-c"]

#RUN ln /DM_HW/DeepMimicCore/third/glew-2.1.0/lib/libGLEW.so.2.1.0 /usr/lib/x86_64-linux-gnu/libGLEW.so.2.1.0

#RUN ln /DM_HW/DeepMimicCore/third/glew-2.1.0/lib/libGLEW.so.2.1 /usr/lib/x86_64-linux-gnu/libGLEW.so.2.1 

RUN pip install "cython<3"
RUN pip install typing-extensions
WORKDIR /DeepMimic_mujoco