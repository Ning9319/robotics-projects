FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions python3-rosdep \
    python3-numpy python3-matplotlib python3-serial python3-pynput \
    python3-pytest ros-jazzy-demo-nodes-py \
    && rm -rf /var/lib/apt/lists/*

RUN printf '\nsource /opt/ros/jazzy/setup.bash\nif [ -f /ws/install/setup.bash ]; then source /ws/install/setup.bash; fi\n' >> /root/.bashrc
ENV MPLBACKEND=Agg
WORKDIR /ws
CMD ["sleep", "infinity"]
