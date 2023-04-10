#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
// 如果没有定义标识符_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_时，就定义标识符
#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16  //最长的类别名，用来做数组
#define OBJ_NUMB_MAX_SIZE 64  //可能是一个图片里最多多少个类别，用来存储每个类别
#define OBJ_CLASS_NUM     80  //一共最多有多少类别
#define NMS_THRESH        0.45// NMS的阈值
#define BOX_THRESH        0.25// BOX的阈值
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;// box的结构体

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t; //一个框的信息 包括类别名字和框 置信度？？

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t; //一张图像的检测结果

//后处理函数
int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

void deinitPostProcess();
#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
