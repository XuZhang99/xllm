include(cc_library)


cc_library(
  NAME
    model_args
  HDRS
    model_args.h
    model_args_factory.h
    qwen3_model_args.h
  DEPS
    :common
)