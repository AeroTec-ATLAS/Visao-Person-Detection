#!/bin/bash

PORT=5600

gst-launch-1.0 -v udpsrc port=$PORT caps="application/x-rtp, media=video, encoding-name=H264" \
! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
