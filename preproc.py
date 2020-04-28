import cv2
import imgs

def scaled_frames(in_path,out_path):
    imgs.transform(in_path,out_path,scale,single_frame=True)

def smooth_frames(in_path,out_path):
    fun=[gauss_helper,scale]
    imgs.transform(in_path,out_path,gauss_helper,single_frame=True)

def scale(img_i):
    dim=(64,64)
    inter=cv2.INTER_CUBIC
    return cv2.resize(img_i,dim,inter)

def gauss_helper(img_i):
    return cv2.GaussianBlur(img_i, (9, 9), 0)

if __name__=="__main__":
    smooth_frames("../MSR/box","gauss_test")