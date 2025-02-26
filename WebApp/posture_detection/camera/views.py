from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from detect.utils import gen_frames

@login_required
def index(request):
    return render(request, "camera/index.html")

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
