# https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html

FROM rocm/pytorch:rocm2.9_ubuntu16.04_py3.6_pytorch
RUN apt update && apt install -y libidn11 libboost-dev rtl-sdr

# Grasp and compile Pytorch
RUN rm -rf ~/pytorch
RUN apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
RUN cd ~ && git clone https://github.com/pytorch/pytorch.git && cd pytorch && git submodule init && git submodule update --init --recursive && git checkout e42af97349274c90cbcbd50aebeb3fa5ee32eea8
RUN cd ~/pytorch && sed -i '/python setup.py install --user/c\python setup.py bdist_wheel' .jenkins/pytorch/build.sh
RUN cd ~/pytorch && sed -i '/python setup.py install/c\python setup.py bdist_wheel' .jenkins/pytorch/build.sh
RUN pip install wheel
RUN cd ~/pytorch && .jenkins/pytorch/build.sh

# Grasp and compile TorchVision
RUN cd ~ && git clone https://github.com/pytorch/vision && cd vision && git checkout 44a5bae933655ed7ff798669a43452b833f9ce01
RUN cd ~/vision && python setup.py bdist_wheel

# Install pyenv
RUN curl https://pyenv.run | bash
ENV PATH=/root/.pyenv/shims:/root/.pyenv/bin:$PATH
RUN echo 'export PATH="~/.pyenv/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN eval "$(pyenv init -)"
RUN pyenv install 3.6.9

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
RUN echo 'source ~/.poetry/env' >> ~/.bashrc
RUN ~/.poetry/bin/poetry config virtualenvs.in-project true