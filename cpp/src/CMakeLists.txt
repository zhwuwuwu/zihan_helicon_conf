# 查找SentencePiece
find_path(SENTENCEPIECE_INCLUDE_DIR 
    NAMES sentencepiece_processor.h
    PATHS ${CMAKE_SOURCE_DIR}/third_party/sentencepiece/include
    NO_DEFAULT_PATH
)

find_library(SENTENCEPIECE_LIBRARY 
    NAMES sentencepiece
    PATHS ${CMAKE_SOURCE_DIR}/third_party/sentencepiece/lib
    NO_DEFAULT_PATH
)

# 查找OpenVINO
find_package(OpenVINO REQUIRED)

# 收集所有源文件
file(GLOB_RECURSE SDK_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

file(GLOB_RECURSE SDK_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/*.h"
)

# 创建DLL
add_library(HeliconSearchSDK SHARED ${SDK_SOURCES} ${SDK_HEADERS})

# 设置DLL导出定义
target_compile_definitions(HeliconSearchSDK PRIVATE HELICON_SDK_EXPORTS)

# 设置包含目录
target_include_directories(HeliconSearchSDK PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/meeting_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/vlm
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/chineseclip
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/stb  # 用于stb_image.h
    ${CMAKE_SOURCE_DIR}/third_party/sentencepiece/include  # 用于sentencepiece
)

# Windows特定设置
if(WIN32)
    set_target_properties(HeliconSearchSDK PROPERTIES
        OUTPUT_NAME "HeliconSearchSDK"
        SUFFIX ".dll"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# 链接库
target_link_libraries(HeliconSearchSDK PRIVATE 
    openvino::runtime
    ${SENTENCEPIECE_LIBRARY}
    ws2_32 
    winmm
)

# C++17 filesystem支持
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(HeliconSearchSDK PRIVATE stdc++fs)
endif()