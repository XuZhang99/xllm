include(cc_library)
include(cc_test)


cc_library(
  NAME 
    processors
  HDRS
    image_processor.h
    clip_image_processor.h
    minicpmv_image_processor.h
    qwen2_vl_image_processor.h
    pywarpper_image_processor.h
    input_processor.h
  SRCS
    image_processor.cpp
    clip_image_processor.cpp
    minicpmv_image_processor.cpp
    qwen2_vl_image_processor.cpp
    pywarpper_image_processor.cpp
  DEPS
    :common
    :layers
    :chat_template
    glog::glog
    torch
)
