if(NOT BUILD_TESTING)
  return()
endif()

set(XLLM_TEST_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${PROJECT_SOURCE_DIR}/xllm
  ${PROJECT_SOURCE_DIR}/xllm/core
)

file(GLOB_RECURSE XLLM_TEST_DIR_CANDIDATES
     LIST_DIRECTORIES true
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*")

foreach(test_dir IN LISTS XLLM_TEST_DIR_CANDIDATES)
  if(IS_DIRECTORY "${test_dir}")
    file(RELATIVE_PATH test_rel_dir "${CMAKE_CURRENT_SOURCE_DIR}" "${test_dir}")
    if(EXISTS "${PROJECT_SOURCE_DIR}/xllm/${test_rel_dir}")
      list(APPEND XLLM_TEST_INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/xllm/${test_rel_dir}")
    endif()
  endif()
endforeach()

list(REMOVE_DUPLICATES XLLM_TEST_INCLUDE_DIRS)
list(APPEND COMMON_INCLUDE_DIRS ${XLLM_TEST_INCLUDE_DIRS})

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
