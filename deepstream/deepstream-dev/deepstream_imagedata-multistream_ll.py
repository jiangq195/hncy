#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys

sys.path.append('../')
import gi
import configparser

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import time
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import numpy as np
import pyds
import cv2
import os
import os.path
from os import path
import shutil
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
import time
# from embedded_plugin.Sensor import Sensor
import re
from minio import Minio
# from minio.error import ResponseError
import datetime
from io import BytesIO

# from nvjpeg import NvJpeg

fps_streams = {}
frame_count = {}
saved_count = {}
global PGIE_CLASS_ID_LL
PGIE_CLASS_ID_LL = 0
# global PGIE_CLASS_ID_CAR
# PGIE_CLASS_ID_CAR = 2
PRE_TIME = 0
time_gap = 10

MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
buffer_obj_maxsize = 100
buffer_obj = []
primary_confidence_thresh = 0.75
second_confidence_thresh = 0.4
edge_rate_x = 0.05
edge_rate_y = 0.08
edge_threshold = 0.5

primary_component_id = 1
second_component_id = 3

# pgie_classes_str = ['E10136_xt', 'E10136_hl',
#                     'E10135_sf', 'E10135_yz', 'E10135_cd',
#                     'E00217_tl',
#                     'E10107_lyj',
#                     'E10207_ybcd', 'E10207_unofficial', 'E10207_official', 'E10207_cbdj',
#                     'E10401_slhc', 'E10401_slc', 'E10401_cp',
#                     'E10406_zdjy', 'E10406_dwjy',
#                     'E00127_xmwgb', 'E00127_dgbtl',
#                     'E00218_zx',
#                     'E00123_jg', 'E00123_ds',
#                     'E00104_qx', 'E00104_qs',
#                     'E10203_hwgg',
#                     'E00221_bdzt']
pgie_classes_str = [
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person',
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person',
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person',
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person',
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person',
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person',
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person',
    'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person'
]
pgie2_classes_str = ['slc', 'slhc'
                     ]

no_reported_classes = ['E10207_official', 'E00123_jg']
tracker_config_path = 'dstest2_tracker_config.txt'


def check_trackerid_exists(tracker_id):
    if tracker_id not in buffer_obj:
        if len(buffer_obj) >= buffer_obj_maxsize:
            buffer_obj.pop(0)
        buffer_obj.append(tracker_id)
        return False
    return True


def put_img_to_minio(image, device_id, detected=False):
    time1 = datetime.datetime.now()
    if not detected:
        path = str(time1.year) + '/' + str(time1.month) + '/' + str(time1.day) + '/' + device_id + '/'
    else:
        path = str(time1.year) + '/' + str(time1.month) + '/' + str(time1.day) + '/' + device_id + '/' + 'detected/'
    times = time1.strftime('%Y-%m-%d %H:%M:%S')
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    image = cv2.imencode('.jpg', image)[1].tostring()
    # image = nj.encode(image)
    image = BytesIO(image)
    image.name = str(timeStamp) + '.jpg'
    try:
        minioClient.put_object(bucket_name='media',
                               object_name=path + image.name,
                               data=image,
                               length=image.getbuffer().nbytes,
                               content_type='image/jpg')
        return image.name
    except:
        print('please check minio server')
        return None


def add_tracker_id(tracker_id):
    if len(buffer_obj) < buffer_obj_maxsize:
        buffer_obj.append(tracker_id)
    else:
        buffer_obj.pop(0)
        buffer_obj.append(tracker_id)


def cross_proportion(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=np.inf) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=np.inf)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    # b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    box1_cross_proportion = inter_area / b1_area

    return box1_cross_proportion


def match_class(primary_class_pre, second_class_pre):
    class_match_flag = False
    cross_threshold = 0
    # The logic list is as follows:
    # if primary_class_pre == 'E10136' and second_class_pre == 'E10401':
    if primary_class_pre == 'E10136_xt' and second_class_pre in ['slc', 'slhc']:
        class_match_flag = True
        cross_threshold = 0.1
    else:
        pass
    return class_match_flag, cross_threshold


# Exclude logical filtering
def cross_logic_processing(primary_obj, second_objs, inter_box):
    flag = True
    primary_class_pre = pgie_classes_str[primary_obj.class_id]
    rect_params = primary_obj.rect_params
    primary_x1y1x2y2 = np.array([[int(rect_params.left),
                                  int(rect_params.top),
                                  int(rect_params.left) + int(rect_params.width),
                                  int(rect_params.top) + int(rect_params.height)]])
    if len(second_objs) > 0:
        for second_obj in second_objs:
            if second_obj.rect_params.width == 0 or second_obj.rect_params.height == 0:
                continue
            inter_rate = filter_edge_box(inter_box, second_obj)
            if second_obj.confidence >= second_confidence_thresh and inter_rate > edge_threshold:
                second_class_pre = pgie2_classes_str[second_obj.class_id][:6]
                class_match_flag, cross_threshold = match_class(primary_class_pre, second_class_pre)
                if class_match_flag:
                    se_rect_params = second_obj.rect_params
                    second_x1y1x2y2 = np.array([[int(se_rect_params.left),
                                                 int(se_rect_params.top),
                                                 int(se_rect_params.left) + int(se_rect_params.width),
                                                 int(se_rect_params.top) + int(se_rect_params.height)]])
                    primary_cross_proportion = cross_proportion(primary_x1y1x2y2, second_x1y1x2y2)
                    if primary_cross_proportion >= cross_threshold:
                        flag = False
                        break
                else:
                    flag = True
            else:
                flag = True
    else:
        flag = True
    return flag


# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and update params for drawing rectangle, object information etc.
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # try:
        #     gps_o = Sensor.getGpsInfo()
        #     g = re.match(r"\[(.*)\,(.*)\]", gps_o)
        #     gps = [str(g[1]), str(g[2])]
        # except Exception as e:
        #     print(gps_o, e)
        #     gps = ['0.0', '0.0']

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        n_frame = cv2.cvtColor(n_frame, cv2.COLOR_BGRA2RGBA)
        frame_copy = np.array(n_frame, copy=True, order='C')
        height, weight = frame_copy.shape[:2]
        inter_box = [int(weight * edge_rate_x), int(height * edge_rate_y), int(weight * (1 - edge_rate_x)),
                     int(height * (1 - edge_rate_y))]

        send_info = False

        detected_info = []
        tracker_ids = []
        ai_model_type = "Unkown"
        image_name = "None"

        primary_objs = []
        second_objs = []
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # if tracker_id < 207360000:
            if obj_meta.unique_component_id == primary_component_id:  # global gie-unique-id
                primary_objs.append(obj_meta)
            elif obj_meta.unique_component_id == second_component_id:
                second_objs.append(obj_meta)
            else:
                print('No such unique_component_id!')
                break
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        print(primary_objs + second_objs)
        now_t = time.time()
        for primary_obj in primary_objs + second_objs:
            tracker_id = primary_obj.object_id
            confidence = primary_obj.confidence
            class_str = pgie_classes_str[primary_obj.class_id]
            if primary_obj.rect_params.width == 0 or primary_obj.rect_params.height == 0:
                continue
            inter_rate = filter_edge_box(inter_box, primary_obj)
            track = True
            if confidence >= primary_confidence_thresh and class_str not in no_reported_classes and inter_rate > edge_threshold:
                flag = cross_logic_processing(primary_obj, second_objs, inter_box)
                # flag = True
                if flag:
                    if tracker_id not in buffer_obj:
                        send_info = True
                        ai_model_type = class_str.split("_")[0]
                        add_tracker_id(tracker_id)
                        track = False

                    tracker_ids.append(tracker_id)
                    frame_copy, data_dict = draw_bounding_boxes(frame_copy, primary_obj, confidence, track, inter_rate)
                    detected_info.append(data_dict)

                else:
                    print('Exclude by cross logic')
                    # test logic
                    if tracker_id not in buffer_obj:
                        send_info = True
                        ai_model_type = class_str.split("_")[0]
                        add_tracker_id(tracker_id)
                        track = False
                    tracker_ids.append(tracker_id)
                    frame_copy, data_dict = draw_bounding_boxes(frame_copy, primary_obj, confidence, track, inter_rate,
                                                                test=True)
                    detected_info.append(data_dict)

        # print("Frame Number=", frame_number, "ll_count=",obj_counter[PGIE_CLASS_ID_LL], send_info, send_image)
        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

        # now_time = int(time.time())
        # global PRE_TIME
        # if now_time - PRE_TIME >= time_gap and minio_status == "on":
        #     put_img_to_minio(n_frame, Sensor.getDeviceId())
        #     PRE_TIME = now_time
        # if send_info and minio_status == "on":
        #     image_name = put_img_to_minio(n_frame, Sensor.getDeviceId(), detected=True)
        json_data = {
            "device_id": "test1",
            "ai_device_id": "test",
            "ai_version": "v1.1.1",
            "ai_model_type": ai_model_type,
            "cam_id": frame_meta.source_id,
            "frame_md5": image_name,
            "frame_time": int(frame_meta.ntp_timestamp / 1000000),
            "detected_time": int(time.time() * 1000),
            "frame_info": {
                "width": weight,
                "height": height
            },
            "detected_info": detected_info,
            "detected_attach": {"trash_overflow": False},
            "attach": {
                "frame_number": frame_meta.frame_num,
                "num_object_meta": frame_meta.num_obj_meta,
                "batch_id": frame_meta.batch_id,
                "gps": ['0.0', '0.0'],
                "speed": ['0.0', '0.0'],
                "gps_original": 0.0
            }
        }
        if send_info and kafka_status == "on":
            # frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGRA2RGBA)
            img_encode = cv2.imencode('.jpg', frame_copy)[1].tostring()
            # img_encode = nj.encode(frame_copy)
            data = (json.dumps(json_data) + '$').encode('utf-8') + img_encode

            try:
                producer.send(kafka_topic, data)
                producer.flush()
                print('--------------send seccess-------------------', tracker_ids)
            except:
                print('send failed, please check connection')

        saved_count["stream_{}".format(frame_meta.pad_index)] += 1
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def filter_edge_box(inter_box, obj_meta):
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    inter_area = [max(inter_box[0], left), max(inter_box[1], top), min(inter_box[2], left + width),
                  min(inter_box[3], top + height)]
    inter_rate = (max((inter_area[2] - inter_area[0]), 0) * max((inter_area[3] - inter_area[1]), 0)) / (width * height)

    return round(inter_rate, 2)


def draw_bounding_boxes(image, obj_meta, confidence, track, inter_rate, test=False):
    if track:
        color_box = (255, 0, 0, 0)
        if test:
            color_box = (0, 0, 0, 0)
    else:
        color_box = (0, 0, 255, 0)
        if test:
            color_box = (0, 255, 0, 0)
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id] + '_' + str(inter_rate)
    image = cv2.rectangle(image, (left, top), (left + width, top + height), color_box, 1, cv2.LINE_4)
    image = cv2.putText(image, obj_name + ',C=' + str(confidence), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color_box, 1)
    return image, {
        "rubbish_id": obj_meta.class_id,
        "class_name": obj_name,
        "score": confidence,
        "x_left": left,
        "y_left": top,
        "x_right": left + width,
        "y_right": top + height
    }


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if is_aarch64() and name.find("nvv4l2decoder") != -1:
        print("Seting bufapi_version\n")
        Object.set_property("bufapi-version", True)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    for i in range(0, len(args)):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
    number_sources = len(args)

    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie2 = Gst.ElementFactory.make("nvinfer", "second-inference")  # second
    if not pgie or not pgie2:
        sys.stderr.write(" Unable to create pgie or pgie2 \n")  # second
    tracker = Gst.ElementFactory.make('nvtracker', 'tracker')
    if not tracker:
        sys.stderr.write('Unable to create tracker \n')
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    if (no_display):
        print("Creating FakeSink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        if not sink:
            sys.stderr.write(" Unable to create fakesink \n")
    else:
        if is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1600)  # 1024
    streammux.set_property('height', 900)  # 768
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "config_infer_primary_yoloV5.txt")
    pgie2.set_property('config-file-path', "config_infer_second_yoloX.txt")  # second
    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
              number_sources, " \n")
        pgie.set_property("batch-size", number_sources)
    pgie2_batch_size = pgie2.get_property("batch-size")
    if (pgie2_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie2_batch_size, " with number of sources ",
              number_sources, " \n")
        pgie2.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH * tiler_columns)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT * tiler_rows)

    sink.set_property("sync", 0)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    config = configparser.ConfigParser()
    config.read(tracker_config_path)
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame':
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    print("Adding elements to Pipeline \n")
    # pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(pgie2)  # second
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if is_aarch64() and not no_display:
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie2)
    pgie2.link(tracker)
    # tracker.link(pgie)     # second
    # pgie.link(nvvidconv1)  # second
    tracker.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64() and not no_display:
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    no_display = False
    with open('terminal_config.json', encoding='utf8') as fp:
        data = json.load(fp)
    kafka_server = data['kafka']['kafka_server']
    kafka_topic = data['kafka']['kafka_topic']
    kafka_status = data['kafka']['on_off']
    minio_server = data['minio']['minio_server']
    minio_access_key = data['minio']['minio_access_key']
    minio_secret_key = data['minio']['minio_secret_key']
    minio_status = data['minio']['on_off']
    producer = KafkaProducer(bootstrap_servers=kafka_server, retries=1)
    minioClient = Minio(minio_server,
                        access_key=minio_access_key,
                        secret_key=minio_secret_key,
                        secure=False)
    streams = [item["pull_stream_url"] for item in data['video'] if item['on_off'] == "on"]
    main(streams)
