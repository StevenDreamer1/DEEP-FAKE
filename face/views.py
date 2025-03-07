from django.shortcuts import render,redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
import os
from face.models import *
from django.contrib import messages
from django.utils.datastructures import MultiValueDictKeyError
from django.contrib.auth import logout

# Create your views here.



def index(request):
    return render(request, 'user/index.html')

def admin_login(request):
    if request.method == "POST":
        username = request.POST.get('name')
        password = request.POST.get('password')
        if username == 'admin' and password == 'admin':
            messages.success(request, 'Login Successful')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid details !')
            return redirect('admin_login')
    return render(request, 'user/Admin-login.html')

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        print(email)
        password = request.POST.get('password')
        print(password)
        try:
            user = User.objects.get(user_email=email)   
            print(user,"userrrrrrrrrrrrrrrr")
            if user.user_password == password:
                request.session['user_id'] = user.user_id
                if user.status == 'Accepted':
                    messages.success(request, 'Login Successful')
                    print("successful")
                    return redirect('user_dashboard')
                else:
                    messages.error(request, 'Your account is not approved yet.')
                    print("failed")
                    return redirect('user_login')

            else:
                messages.error(request, 'Incorrect Password')
                return redirect('user_login')
        except User.DoesNotExist:
            messages.error(request, 'Invalid Login Details')
            return redirect('user_login')
    return render(request, 'user/user-login.html')



def user_register(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        password = request.POST.get('password')
        location = request.POST.get('address')
        profile = request.FILES.get('profile')

        # Add print statements to check values
        print("Name:", name)
        print("Email:", email)
        print("Phone:", phone)
        print("Password:", password)
        print("Location:", location)
        print("Profile:", profile)

        try:
            User.objects.get(user_email = email)
            messages.info(request, 'Email Already Exists!')
            return redirect('register')
        except:
            user = User.objects.create(user_name=name, user_email=email, user_phone=phone, user_profile=profile, user_password=password, user_location=location)
            messages.success(request, 'Registerd Successfully !')
            print(user)
            user_id_new = request.session['user_id'] = user.user_id
            print(user_id_new)
            return redirect('user_login')

    return render(request, 'user/user-register.html')


def about(request):
    return render(request, 'user/about.html')

def contact(request):
    return render(request, 'user/contact.html')


def user_dashboard(request):
    return render(request, 'user/user-dashboard.html')


from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


model_path = 'model_dk.h5'
model = load_model(model_path)

def prediction(path):
  img = image.load_img(path, target_size=(224, 224))
  i = image.img_to_array(img)
  i = np.expand_dims(i, axis=0)
  img = preprocess_input(i)
  pred = np.argmax(model.predict(img), axis=1)
  print(f"the image belongs to {[pred[0]]}")
  return pred[0]


from django.conf import settings

def image_detection(request):
    context = {}  
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        print(uploaded_file, "Uploded file is")

        temp_file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

        with open(temp_file_path, 'wb') as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)

        print("Uploaded file path:", temp_file_path)

        result = prediction(temp_file_path)
        print(result,"Result ----------------------")
        context = {'result': result}

    return render(request, 'user/image-detection.html', context)






import os
from threading import Thread
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Import Queue from queue module
from queue import Queue

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained model
model = load_model(r"model_dk.h5")

# Define your class labels (e.g., ['Fake', 'Real'])
class_labels = ['Fake', 'Real']

def detect_face(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Return the coordinates of the first detected face (assuming there is only one face)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return (x, y, w, h)
    else:
        return None

def predict_and_display(frame, model, class_labels, results_queue):
    # Detect face
    face_coords = detect_face(frame)

    if face_coords is not None:
        x, y, w, h = face_coords

        # Resize the frame to the target size expected by the model
        face_roi = frame[y:y+h, x:x+w]
        img = cv2.resize(face_roi, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        # Put the result in the queue
        results_queue.put(predicted_class_label)


def capture_frames(video_path, results_queue):
    # Open a connection to the video file
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        predict_and_display(frame, model, class_labels, results_queue)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()


def video_detection(request):
    context = {}

    if request.method == 'POST' and 'video' in request.FILES:
        uploaded_video = request.FILES['video']
        print(uploaded_video, "nnnnnnnnnnnnnnnnnnnnnnnnnn")

        video_directory = os.path.join(settings.MEDIA_ROOT, 'videos')
        os.makedirs(video_directory, exist_ok=True)

        video_name = uploaded_video.name
        video_path = os.path.join(video_directory, video_name)

        with open(video_path, 'wb') as video_file:
            for chunk in uploaded_video.chunks():
                video_file.write(chunk)

        print("Uploaded video path:", video_path)

        results_queue = Queue()

        thread = Thread(target=capture_frames, args=(video_path, results_queue))
        thread.start()

        thread.join()

        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        fake_count = results.count('Fake')
        real_count = results.count('Real')

        if fake_count > 30:
            overall_result = 'Fake'
        elif fake_count < 10 and real_count < 10:
            overall_result = 'No Face Detected' 
            messages.info(request,"No Face Is Detected In Video")              
        else:
            overall_result = 'Real'

        print(f"Overall Result: {overall_result}")
        print(f"Fake Count: {fake_count}, Real Count: {real_count}")

        context = {'result': overall_result}
        

    return render(request, 'user/video-detection.html', context)





def profile(request):
    user_id  = request.session['user_id']
    print(user_id)
    user = User.objects.get(pk= user_id)
    if request.method == "POST":
        name = request.POST.get('name')
        # email = request.POST.get('email')
        phone = request.POST.get('phone')
        try:
            profile = request.FILES['profile']
            user.user_profile = profile
        except MultiValueDictKeyError:
            profile = user.user_profile
        password = request.POST.get('password')
        location = request.POST.get('location')
        user.user_name = name
        # user.user_email = email
        user.user_phone = phone
        user.user_password = password
        user.user_location = location
        user.save()
        messages.success(request , 'updated succesfully!')
        return redirect('profile')
    return render(request, 'user/profile.html',{'user':user})



def user_logout(request):
    logout(request)
    messages.info(request,"You are logged out !")
    return redirect('user_login')