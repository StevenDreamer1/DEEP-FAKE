from django.db import models

class Student(models.Model):
    sid = models.CharField(max_length=50)
    sname = models.CharField(max_length=55)
    email = models.EmailField(max_length=55)
    password = models.CharField(max_length=55)
    mno = models.CharField(max_length=11)


class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_name=models.CharField(max_length=50)
    user_email = models.EmailField(max_length=50)
    user_password = models.CharField(max_length=50)
    user_phone = models.CharField(max_length=50)
    user_location = models.CharField(max_length=50,default='Unknown')
    user_profile = models.ImageField(upload_to='images/user')
    status = models.CharField(max_length=15,default='Accepted')
  
  

    class Meta:
        db_table = 'User_details'



# Create your models here.
