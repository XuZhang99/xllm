include(cc_binary)

include_directories(.)

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
add_subdirectory(proto)
add_subdirectory(processors)
add_subdirectory(pybind)
add_subdirectory(server)

cc_binary(
  NAME
    xllm
  SRCS
    xllm.cpp
  DEPS
    :flags
    :xllm_server
    :master
    absl::strings
    Boost::serialization
    gflags::gflags
    glog::glog
    Folly::folly
    nlohmann_json::nlohmann_json
    $<$<BOOL:${USE_NPU}>:c_sec>
)

if(USE_MSPTI)
  target_link_libraries(xllm PUBLIC mspti)
endif()

# install xllm binary
install(TARGETS xllm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# Install all dependencies for xllm
install(CODE [[
    file(GET_RUNTIME_DEPENDENCIES
          RESOLVED_DEPENDENCIES_VAR DEPENDENCIES
          UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPENDENCIES
          EXECUTABLES $<TARGET_FILE:xllm>)

    file(INSTALL
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
          FILES ${DEPENDENCIES}
          FOLLOW_SYMLINK_CHAIN)

    # This should not be possible, but error out when a dependency cannot
    # be resolved.
    list(LENGTH UNRESOLVED_DEPENDENCIES UNRESOLVED_LENGTH)
    if(${UNRESOLVED_LENGTH} GREATER 0)
        message(FATAL_ERROR "Unresolved dependencies: ${UNRESOLVED_DEPENDENCIES}")
    endif()
]])

