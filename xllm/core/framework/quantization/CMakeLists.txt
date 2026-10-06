include(cc_library)

cc_library(
  NAME
    quantization
  HDRS
    quant_args.h
  DEPS
    :common
    :state_dict
    glog::glog
)
