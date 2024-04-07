#----------------------------------------------------#
# Single image prediction, camera detection and FPS testing functions
# Integrate into a py file and change the mode by specifying mode.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet_ONNX, Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    # If you want to change the colors of the corresponding class, go to the __init__ function and change self.colors
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    # mode Specifies the mode of the test:
    # 'predict' indicates a single image prediction. If you want to modify the prediction process, such as saving images, capturing objects, etc., you can first read the detailed comments below
    # 'video' indicates video detection. Cameras or videos can be used for detection. See the comments below for details.
    # 'fps' means test fps, the image used is in img's street.jpg, see the comments below for details.
    # 'dir_predict' means to traverse the folder for detection and saving. By default, traverse the img folder and save the img_out folder. See the comments below for details.
    # 'export_onnx' means that exporting the model as onnx requires pytorch1.7.1 or more.
    # 'predict_onnx' means Unet_ONNX for prediction using the derived onnx model, with the modification of the relevant parameters around the unet.py_346 line
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    # count Specifies whether the pixel count (i.e. area) and scale calculation of the target are performed
    # name_classes category, same as in json_to_dataset, used to print category and quantity
    #
    # count and name_classes are only valid if mode='predict'
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    #----------------------------------------------------------------------------------------------------------#
    # video_path Specifies the path of the video. video_path=0 indicates that the camera is detected
    # If you want to detect the video, set it as video_path = "xxx.mp4", which means read and take out the xxx.mp4 file in the root directory.
    # video_save_path indicates the path where the video is saved. If video_save_path="", the video is not saved
    # video_fps fps used to save the video
    #
    # video_path, video_save_path, and video_fps are only valid if mode='video'
    # When saving a video, you need ctrl+c to exit or run until the last frame to complete the save step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    # test_interval specifies the number of image detections when measuring fps. Theoretically, the larger the test_interval, the more accurate the fps.
    # fps_image_path is used to specify fps images for testing
    #
    # test_interval and fps_image_path only work in mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    # dir_origin_path specifies the folder path of the images to be detected
    # dir_save_path specifies the path to save the detected image
    #
    # dir_origin_path and dir_save_path are only valid if mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    # simplify with Simplify onnx
    # onnx_save_path specifies the path to save onnx
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        '''
        predict.py has several caveats
        1, the code can not be directly for batch prediction, if you want to batch prediction, you can use os.listdir() to traverse the folder, 
        using image.open to open the Image file for prediction.
        For the specific process, refer to get_miou_prediction.py, where the traversal is implemented.
        2, if you want to save, use r_image.save("img.jpg") to save.
        3, if you want the original and split maps not to mix, you can set the blend parameter to False.
        4, if you want to obtain the corresponding area according to the mask, you can refer to the detect_image function, 
        the use of the prediction result of the drawing part, determine the type of each pixel, and then obtain the corresponding part according to the type.
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Camera (video) error, please note correct installation (video path)。")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(unet.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)
                
    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
