FROM nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-base


RUN mkdir /app
COPY ./librdkafka /app/librdkafka
COPY ./gst-python /app/gst-python
COPY ./requirements.txt /app/requirements.txt
COPY ./deepstream_python_apps /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps
# COPY ./jetson_multimedia_api /usr/src/jetson_multimedia_api
RUN apt-get update \
    && apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstrtspserver-1.0-dev gstreamer1.0-rtsp libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libx11-dev libjson-glib-dev libssl1.0.0  libgstrtspserver-1.0-0 libjansson4 build-essential vim python3-pip python-gi-dev python3-dev libpython3-dev -y \
    && cd /app/librdkafka && git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a && ./configure && make && make install \
    && cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-5.0/lib \
    && export GST_LIBS="-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0" \
    && export GST_CFLAGS="-pthread -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include" \
    && cd /app/gst-python && git checkout 1a8f48a \
    && ./autogen.sh PYTHON=python3 && ./configure PYTHON=python3 \
    && make && make install && cd /app \
    && pip3 install --upgrade pip && pip3 install -r requirements.txt -i https://pypi.douban.com/simple \
    && rm -rf /app

ENV LD_PRELOAD="/opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream-imagedata-multistream/class_ll/libmyplugins.so"
WORKDIR /opt/nvidia/deepstream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream-imagedata-multistream
ENTRYPOINT ["python3", "deepstream_imagedata-multistream_test2.py"]
