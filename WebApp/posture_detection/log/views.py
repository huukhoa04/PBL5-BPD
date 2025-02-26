from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages

def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("index")  
        else:
            messages.error(request, "Sai thông tin đăng nhập!")
    return render(request, "log/login.html")

def register_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        confirm_password = request.POST["confirm_password"]
        if password == confirm_password:
            if not User.objects.filter(username=username).exists():
                user = User.objects.create_user(username=username, password=password)
                user.save()
                messages.success(request, "Đăng ký thành công!")
                return redirect("login")
            else:
                messages.error(request, "Tên người dùng đã tồn tại!")
        else:
            messages.error(request, "Mật khẩu không khớp!")
    return render(request, "log/register.html")

def logout_view(request):
    logout(request)
    return redirect("login")
