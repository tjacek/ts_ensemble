import cv2
import imgs

def scaled_frames(in_path,out_path):
    def scale_helper(img_i):
        dim=(64,64)
        inter=cv2.INTER_CUBIC
        return cv2.resize(img_i,dim,inter)
    imgs.transform(in_path,out_path,scale_helper,single_frame=True)

if __name__=="__main__":
    scaled_frames("../MSR/box","scale_test")