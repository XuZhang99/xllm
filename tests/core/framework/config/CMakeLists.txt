include(cc_test)

cc_test(
  NAME
    config_json_test
  SRCS
    config_json_test.cpp
  DEPS
    :framework_config
    util
    GTest::gtest_main
)
