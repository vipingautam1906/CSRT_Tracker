import imutils
import argparse
import time
import cv2
from imutils.video import VideoStream
from imutils.video import FPS

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, help='path to input video file')  # if we skip --video then it will use webcam
ap.add_argument('-t', '--tracker', type=str ,default='csrt', help='OpenCV object track type')
args = vars(ap.parse_args())

opencv_oject_tracker = {
    'csrt': cv2.TrackerCSRT_create,
    'kcf': cv2.TrackerKCF_create
}
tracker = opencv_oject_tracker[args['tracker']]()
init_bb = None

if not args.get('video', False):
    print("Starting video stream..")
    video_stream = VideoStream(src=0).start()
    time.sleep(1.0)
else:
    video_stream = cv2.VideoCapture((args['video']))

fps = None

while True:
    current_frame = video_stream.read()
    current_frame = current_frame[1] if args.get('video', False) else current_frame
    if current_frame is None:
        break

    current_frame = imutils.resize(current_frame, width=700)
    (H, W) = current_frame.shape[:2]

    # case when object is already selected
    if init_bb is not None:
        (success, csrt_box) = tracker.update(current_frame)
        if success == False:
            print("Object has left the FOV")
        if success:
            x, y, w, h = [ int(x) for x in csrt_box ]
            cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        fps.update()
        fps.stop()

        # info which we will show in each frame
        info = [
            ("Tracker", args['tracker']),
            ('success', 'Yes' if success else 'No'),
            ('FPS', '{:.2f}'.format(fps.fps()))
        ]

        # showing info on each frame
        for (i, (k, v)) in enumerate(info):
            text = '{}: {}'.format(k, v)
            cv2.putText(current_frame,text,(10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Frame', current_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # selecting th bb box of the object that we wana track
        # press enter when we have selected ROI
        init_bb = cv2.selectROI('Frame', current_frame, fromCenter=False, showCrosshair=True)
        # starting tracker with using ROI that we selected
        tracker.init(current_frame, init_bb)
        # FPS throughput estimator
        fps = FPS().start()

    elif key == ord('q'):
        break

if not args.get('video', False):
    video_stream.stop()

else:
    video_stream.release()

cv2.destroyAllWindows()

