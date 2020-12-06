FROM chenrocks/uniter
RUN pip install loguru fire easydict tensorboardX==2.1 msgpack-numpy==0.4.6.1 dataclasses
RUN pip install -U ray
RUN apt-get update \
    && apt-get install apt-transport-https ca-certificates htop
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get install -y apt-transport-https ca-certificates \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update \
    && apt-get install -y google-cloud-sdk
RUN sed -i "1iforce_color_prompt=true" ~/.bashrc
# RUN ls
WORKDIR /src
