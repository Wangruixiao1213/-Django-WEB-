from django.shortcuts import render,redirect
from django.shortcuts import HttpResponse
from django.shortcuts import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
import pymysql
import os
import sys
import smtplib
from email.mime.text import MIMEText
import random as r

emailUser = 'abv@qq.com'
verifi = '100000'
login_status = False


def videoOn(request):
    os.system("python detect_mask_video.py")
    return HttpResponseRedirect('/index/')

def logout(request):
    global login_status
    login_status = False
    print('1')
    return HttpResponseRedirect('/login/')

def login(request):
    return render(request, 'login.html')

def register(request):
    return render(request,'register.html')

def forget(request):
    return render(request,'forget.html')

def index(request):
    if login_status == True:
        return render(request,'index.html')
    else:
        return HttpResponseRedirect('/login/')

def send(request):
    a = request.GET
    eMail = a.get('email')
    global emailUser
    emailUser = eMail
    global verifi
    verifi = str(r.randrange(100000,999999))
    sendmail(eMail,verifi)
    return HttpResponse('验证码已发送')

def find(request):
    a = request.GET
    verifi_user = a.get('verification')
    if verifi_user == verifi:
        db = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='maskdetector')
        cursor = db.cursor()
        sqlFind = 'select * from user'
        cursor.execute(sqlFind)
        all_users = cursor.fetchall()
        username = ''
        passwd = ''
        i = 0
        has_regiter = 0
        while i < len(all_users):
            if emailUser in all_users[i]:
                ##表示该邮箱已经存在
                username = all_users[i][0]
                passwd = all_users[i][1]
                has_regiter = 1
            i += 1
        if has_regiter == 0:
            return HttpResponse("该邮箱未注册")
        elif has_regiter == 1:
            cursor.close()
            db.close()
            information = 'your username:' + username + '\n your password:' + passwd
            return  HttpResponse(information)
    else:
        return HttpResponse('验证码错误')

# Create your views here.
# 定义一个函数，用来保存注册的数据
def save(request):
    has_regiter = 0#用来记录当前账号是否已存在，0：不存在 1：已存在
    a = request.GET#获取get()请求
    #print(a)
    #通过get()请求获取前段提交的数据
    userName = a.get('username')
    passWord = a.get('password')
    eMail = a.get('email')

    #print(userName,passWord)
    #连接数据库
    db = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='maskdetector')
    #创建游标
    cursor = db.cursor()
    #SQL语句
    sqlFind = 'select * from user'
    #执行SQL语句
    cursor.execute(sqlFind)
    #查询到所有的数据存储到all_users中
    all_users = cursor.fetchall()
    i = 0
    while i < len(all_users):
        if userName in all_users[i]:
            ##表示该账号已经存在
            has_regiter = 1
        i += 1
    if has_regiter == 0:
        # 将用户名与密码插入到数据库中
        sqlInsertRegister = 'insert into user(username,password,email) values(%s,%s,%s)'
        cursor.execute(sqlInsertRegister,(userName,passWord,eMail))
        db.commit()
        cursor.close()
        db.close()
        return  HttpResponseRedirect('/login/')

    else:

        cursor.close()
        db.close()
        return HttpResponse('该账号已存在')

def query(request):
    a = request.GET
    userName = a.get('username')
    passWord = a.get('password')
    user_tup = (userName,passWord)
    db = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='maskdetector')
    cursor = db.cursor()
    sqlFind = 'select * from user'
    cursor.execute(sqlFind)
    all_users = cursor.fetchall()
    cursor.close()
    db.close()
    has_user = 0
    i = 0
    while i < len(all_users):
        if user_tup[0] == all_users[i][0] and user_tup[1] == all_users[i][1]:
            has_user = 1
            break
        i += 1
    if has_user == 1:
        global login_status
        login_status = True
        request.session.set_expiry(0)
        return  HttpResponseRedirect('/index/')
    else:
        return HttpResponse('用户名或密码有误')

def sendmail(msg_to,verification):
    msg_from = '847573508@qq.com'
    passwd = 'rosddzicgslubaij'
    # msg_to = '2019141460130@stu.scu.edu.cn'

    subject = "找回账号密码"
    content = "你的验证码为" + verification
    msg = MIMEText(content)
    msg['Subject']= subject
    msg['From'] = msg_from
    msg['To'] = msg_to

    s = smtplib.SMTP_SSL('smtp.qq.com',465)
    s.login(msg_from,passwd)
    s.sendmail(msg_from,msg_to,msg.as_string())
    print("succeed")
