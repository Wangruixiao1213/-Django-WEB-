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