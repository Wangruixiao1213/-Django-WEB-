from django.db import models
class User(models.Model):

    # 其实没有使用 用的是mysql
    id = models.IntegerField(primary_key=True)
    username = models.CharField(max_length=32)
    password = models.CharField(max_length=32)
    email = models.EmailField()
# Create your models here.
