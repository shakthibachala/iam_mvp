pybind11_add_module(_iam_core core_bindings.cpp)
target_link_libraries(_iam_core PRIVATE iam_core)
install(TARGETS _iam_core DESTINATION ${CMAKE_SOURCE_DIR}/python/iam)