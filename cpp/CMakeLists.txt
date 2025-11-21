cmake_minimum_required(VERSION 3.16)
project(HeliconSearchSDK VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器选项
if(MSVC)
    # Windows MSVC编译器选项
    add_compile_options(/W3)
    add_compile_options(/utf-8)  # 支持UTF-8编码
    # 设置运行时库为多线程DLL
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
else()
    # GCC/Clang编译器选项
    add_compile_options(-Wall -Wextra)
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# 添加子目录
add_subdirectory(src)
add_subdirectory(test)


find_package(Threads REQUIRED)

# 如果使用filesystem库（C++17之前的编译器可能需要）
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(HeliconSearchSDK PRIVATE stdc++fs)
endif()


# 设置启动项目（Visual Studio）
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SDKTest)
endif()