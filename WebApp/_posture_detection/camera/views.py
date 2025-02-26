from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.contrib.auth.decorators import login_required
import cv2

@login_required
def index(request):
    return render(request, "camera/index.html")

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
def gen_frames():
    cap = cv2.VideoCapture("http://192.168.100.126:81/stream")
    if not cap.isOpened():
        print("Không thể mở luồng video. Kiểm tra lại URL!")
    else:
        print("Luồng video hoạt động bình thường.")
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')