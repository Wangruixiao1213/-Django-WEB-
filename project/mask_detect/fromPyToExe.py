from PyInstaller.__main__ import run
if __name__ == '__main__':
    # 尝试生成exe文件 但尝试了多种方法 生成后的exe文件还是无法运行 故选择放弃
    opts = ['detect_mask_video.py','-D']
    run(opts)