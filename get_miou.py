import os

from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

'''
The following points should be noted in indicator evaluation:
1, the file generated gray scale map, because the value is relatively small, according to the JPG form of the map is not displayed, so it is normal to see approximately all black.
2. The file calculates the miou of the verification set. Currently, the library uses the test set as the verification set and does not divide the test set separately
3. Only models trained according to VOC format data can use this file to perform miou calculations.
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    # miou_mode is used to specify what the file evaluates at runtime
    # miou_mode 0 indicates the entire miou calculation process, including obtaining prediction results and calculating miou.
    # miou_mode of 1 means that only the prediction result is obtained.
    # miou_mode 2 means that only miou is calculated.
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    # Number_classes +1, such as 2+1
    #------------------------------#
    num_classes     = 21
    #--------------------------------------------#
    # The type of distinction is the same as in json_to_dataset
    #--------------------------------------------#
    name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    #-------------------------------------------------------#
    # Point to the folder where the VOC dataset is located
    # Default to the VOC dataset in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)