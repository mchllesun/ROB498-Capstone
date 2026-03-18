#!/usr/bin/env python3
"""
Standalone YOLO inference benchmark on live IMX219 camera feed.
No ROS, no pose, no mavros. Just camera -> YOLO -> print results.

Usage:
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 yolo_bench.py
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 yolo_bench.py --fp16
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 yolo_bench.py --device cpu
"""

import argparse
import time
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from ultralytics import YOLO


PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=10/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw,width=640,height=640,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink name=appsink emit-signals=true max-buffers=1 drop=true"
)


class Camera:
    def __init__(self):
        Gst.init(None)
        self._frame = None
        pipeline = Gst.parse_launch(PIPELINE)
        appsink = pipeline.get_by_name('appsink')
        appsink.set_property('emit-signals', True)
        appsink.connect('new-sample', self._on_sample)
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to open camera pipeline:\n{PIPELINE}")
        self._pipeline = pipeline
        print("Camera opened. Waiting for first frame...")

    def _on_sample(self, appsink):
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        h, w = caps.get_value('height'), caps.get_value('width')
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if ok:
            self._frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 3).copy()
            buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def read(self):
        return self._frame

    def close(self):
        self._pipeline.set_state(Gst.State.NULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/jetson/ROB498-Capstone/install/umbrella/share/umbrella/models/yolo26n.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--conf', type=float, default=0.35)
    args = parser.parse_args()

    print(f"\n=== YOLO Benchmark ===")
    print(f"  model:  {args.model}")
    print(f"  device: {args.device}")
    print(f"  fp16:   {args.fp16}")
    print(f"  imgsz:  {args.imgsz}")
    print(f"  conf:   {args.conf}")
    print()

    print("Loading YOLO model...")
    model = YOLO(args.model)
    print("Model loaded.\n")

    cam = Camera()

    # Wait for first frame
    while cam.read() is None:
        time.sleep(0.05)
    print("First frame received.\n")

    print("Running warmup inference (may take 10-30s on first CUDA call)...")
    t0 = time.time()
    frame = cam.read()
    model.predict(frame, device=args.device, half=args.fp16, imgsz=args.imgsz,
                  verbose=False, conf=args.conf)
    warmup_ms = (time.time() - t0) * 1000
    print(f"Warmup done in {warmup_ms:.0f} ms\n")
    print("=" * 50)
    print(f"{'Frame':>6}  {'Latency':>10}  {'FPS':>6}  {'Detections'}")
    print("=" * 50)

    frame_count = 0
    latencies = []

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            results = model.predict(frame, device=args.device, half=args.fp16,
                                    imgsz=args.imgsz, verbose=False, conf=args.conf)
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)
            frame_count += 1

            # Find the single highest-confidence detection across all classes
            best_cls, best_conf = '—', 0.0
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    conf = float(box.conf.item())
                    if conf > best_conf:
                        best_conf = conf
                        best_cls = results[0].names[int(box.cls.item())]

            det_str = f"{best_cls} {best_conf:.2f}" if best_cls != '—' else '—'
            fps = 1000.0 / latency_ms
            print(f"{frame_count:>6}  {latency_ms:>8.1f}ms  {fps:>5.1f}  {det_str}")

            # Print rolling stats every 30 frames
            if frame_count % 30 == 0:
                avg = sum(latencies[-30:]) / 30
                mn  = min(latencies[-30:])
                mx  = max(latencies[-30:])
                print(f"\n--- Last 30 frames: avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms  avg_fps={1000/avg:.1f} ---\n")

    except KeyboardInterrupt:
        print("\n\nStopped.")
        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"Total frames: {frame_count}")
            print(f"Avg latency:  {avg:.1f} ms  ({1000/avg:.1f} FPS)")
            print(f"Min latency:  {min(latencies):.1f} ms")
            print(f"Max latency:  {max(latencies):.1f} ms")
    finally:
        cam.close()


if __name__ == '__main__':
    main()
