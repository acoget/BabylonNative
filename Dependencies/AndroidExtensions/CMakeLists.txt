set(SOURCES
    "Include/AndroidExtensions/Globals.h"
    "Include/AndroidExtensions/JavaWrappers.h"
    "Source/Globals.cpp"
    "Source/JavaWrappers.cpp")

add_library(AndroidExtensions ${SOURCES})

target_link_to_dependencies(AndroidExtensions 
    PUBLIC arcana)

target_include_directories(AndroidExtensions PUBLIC "Include")
