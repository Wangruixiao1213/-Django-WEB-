import os
# 通过start来启动
if __name__ == '__main__':
    runCmd = 'python manage.py runserver 0.0.0.0:8000'
    # Quit the server with CTRL-BREAK  so where is break?
    os.system(runCmd)