
import cv2
import numpy as np

from rknn.api import RKNN
import os
# 将onnx模型转换为rknn模型
if __name__ == '__main__':

    platform = 'rk3566'
    exp = 'yolov5s'
    Width = 640 #图片输出的大小
    Height = 640
    MODEL_PATH = './onnx_models/yolov5s_rm_transpose.onnx' #要转换的模型路径
    NEED_BUILD_MODEL = True
    # NEED_BUILD_MODEL = False
    im_file = './dog_bike_car_640x640.jpg'

    # Create RKNN object
    rknn = RKNN() #创建RKNN对象，初始化RKNN SDK环境

    OUT_DIR = "rknn_models" #输出目录
    RKNN_MODEL_PATH = './{}/{}_rm_transpose.rknn'.format(OUT_DIR,exp+'-'+str(Width)+'-'+str(Height))
    if NEED_BUILD_MODEL:
        DATASET = './dataset.txt' #数据集目录
        #调用config接口设置模型的预处理参数
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
        # Load model
        print('--> Loading model')
        #调用load接口，调用原始的模型
        ret = rknn.load_onnx(MODEL_PATH)
        if ret != 0: #判断模型是否加载成功
            print('load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        #调用build构建RKNN模型
        ret = rknn.build(do_quantization=True, dataset=DATASET)
        if ret != 0: #判断是否转换成功
            print('build model failed.')
            exit(ret)
        print('done')

        # Export rknn model
        # 导出RKNN模型
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
        print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export rknn model failed.')
            exit(ret)
        print('done')
    else:
        ret = rknn.load_rknn(RKNN_MODEL_PATH)

    rknn.release()#释放对象

