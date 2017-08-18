
# location for fasttext source
set(SRC_PATH "src")

set(fasttext_SRC
        WordEmbedder.cc
        Classifier.cc
        Interop.cc
        ${SRC_PATH}/args.cc
        ${SRC_PATH}/dictionary.cc
        ${SRC_PATH}/fasttext.cc
        ${SRC_PATH}/matrix.cc
        ${SRC_PATH}/model.cc
        ${SRC_PATH}/productquantizer.cc
        ${SRC_PATH}/qmatrix.cc
        ${SRC_PATH}/utils.cc
        ${SRC_PATH}/vector.cc
        ${SRC_PATH}/main.cc
        )

add_executable(fasttext ${SRC_PATH}/main.cc)

set_target_properties(exe PROPERTIES
        OUTPUT_NAME fasttext
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
        COMPILE_FLAGS "-pthread "
        )


#target_link_libraries(exe core)