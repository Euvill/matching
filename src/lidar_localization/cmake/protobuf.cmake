find_package (Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

list(APPEND ALL_TARGET_LIBRARIES ${Protobuf_LIBRARIES})
