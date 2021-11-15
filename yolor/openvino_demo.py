"""
 Copyright (C) 2018-2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time
import numpy as np

import ngraph
import cv2
from openvino.inference_engine import IENetwork, IECore
from utils import non_max_suppression, scale_coords


logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    # args.add_argument("-at", "--architecture_type", help='Required. Specify model\' architecture type.',
    #                   type=str, required=True, choices=('yolov3', 'yolov4', 'yolov5', 'yolov4-p5', 'yolov4-p6', 'yolov4-p7'))
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
    args.add_argument("--save", help="Optional. Save the output", action="store_true")
    return parser


def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


def main():
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
    #if "CPU" in args.device:
    #    supported_layers = ie.query_network(net, "CPU")
    #    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    #    if len(not_supported_layers) != 0:
    #        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
    #                  format(args.device, ', '.join(not_supported_layers)))
    #        log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
    #                  "or --cpu_extension command line argument")
    #        sys.exit(1)
    #
    #assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
    input_blob = next(iter(net.inputs))

    #  Defaulf batch_size is 1
    net.batch_size = 1

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    input_stream = 0 if args.input == "cam" else args.input

    is_async_mode = True
    cap = cv2.VideoCapture(input_stream)
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    wait_key_code = 1

    # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
    if number_input_frames != 1:
        ret, frame = cap.read()
    else:
        is_async_mode = False
        wait_key_code = 0

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)

    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")
    while cap.isOpened():
        # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
        # in the regular mode, we capture frame to the CURRENT infer request
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()

        if not ret:
            break

        if is_async_mode:
            request_id = next_request_id
            in_frame = letterbox(frame, (w, h))
        else:
            request_id = cur_request_id
            in_frame = letterbox(frame, (w, h))
        # resize input_frame to network size
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))

        # Start inference
        start_time = time()
        exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})

        # Collecting object detection results
        objects = list()
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            det_time = time() - start_time
            output = exec_net.requests[cur_request_id].outputs
            start_time = time()

            pred = output['output']
            pred = non_max_suppression(output['output'],
                                        conf_thres=args.prob_threshold,
                                        iou_thres=args.iou_threshold)
            
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(in_frame.shape[2:], det[:, :4], frame.shape).round()
                
                # Write results
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = list(map(int, xyxy))
                    
                    color = (int(min(int(cls) * 12.5, 255)),
                             min(int(cls) * 7, 255),
                             min(int(cls) * 5, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, str(int(cls)), (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, color, 1)
            parsing_time = time() - start_time
        
        # Draw performance stats over frame
        inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
            "Inference time: {:.3f} ms".format(det_time * 1e3)
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
        async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
            "Async mode is off. Processing request {}".format(cur_request_id)
        parsing_message = "YOLO parsing time is {:.3f} ms".format(parsing_time * 1e3)

        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        cv2.putText(frame, async_mode_message, (10, int(frame.shape[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (10, 10, 200), 1)
        cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        start_time = time()
        if not args.no_show:
            cv2.imshow("DetectionResults", frame)
        
        if args.save:
            cv2.imwrite('output.jpg', frame)
        render_time = time() - start_time

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        if not args.no_show:
            key = cv2.waitKey(wait_key_code)
    
            # ESC key
            if key == 27:
                break
            # Tab key
            if key == 9:
                exec_net.requests[cur_request_id].wait()
                is_async_mode = not is_async_mode
                log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
