FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /workspaces
# COPY src .

RUN apt update
RUN apt install -y  \    
    git             \
    libxcb1         \
    libgl1          \
    libglib2.0-0    \
    python3.12-venv

RUN python3 -m venv venv 


RUN venv/bin/pip install ultralytics opencv-python scikit-learn tqdm fastapi streamlit
RUN echo "source /workspaces/venv/bin/activate" >> ~/.bashrc

RUN rm -rf /var/lib/apt/lists/*


CMD ["sleep", "infinity"]


