# 测试程序源文件
set(TEST_SOURCES 
    SDKTest.cpp
)

# 创建可执行文件
add_executable(SDKTest ${TEST_SOURCES})

# 链接SDK库
target_link_libraries(SDKTest PRIVATE HeliconSearchSDK)

# 设置包含目录 - 确保所有路径正确
target_include_directories(SDKTest PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/meeting_manager
    ${CMAKE_SOURCE_DIR}/src/modules
    ${CMAKE_SOURCE_DIR}/src/modules/vlm
)

# 设置依赖关系
add_dependencies(SDKTest HeliconSearchSDK)

# Windows特定设置
if(WIN32)
    add_custom_command(TARGET SDKTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:HeliconSearchSDK>
        $<TARGET_FILE_DIR:SDKTest>
        COMMENT "Copying HeliconSearchSDK.dll"
    )
endif()

# 复制 OpenVINO DLLs

message(STATUS "OpenVINO_DIR: ${OpenVINO_DIR}")
set(OPENVINO_BIN_PATH "${OpenVINO_DIR}/../../runtime/bin/intel64/Release")
message(STATUS "Using OpenVINO Release DLLs from: ${OPENVINO_BIN_PATH}")

if(WIN32)
    add_custom_command(TARGET SDKTest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OpenVINO_DIR}/../../runtime/bin/intel64/Release"
        "$<TARGET_FILE_DIR:SDKTest>"
        COMMENT "Copying OpenVINO DLLs"
    )
endif()