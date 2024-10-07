#project(mumaxplus)

add_library(physics STATIC 
    afmexchange.cu
    afmexchange.hpp
    anisotropy.cu 
    anisotropy.hpp 
    antiferromagnet.cpp
    antiferromagnet.hpp
    antiferromagnetquantity.cpp
    antiferromagnetquantity.hpp
    conductivitytensor.cu
    conductivitytensor.hpp
    constants.hpp
    demag.cpp
    demag.hpp
    dmi.cu
    dmi.hpp
    dmitensor.cpp
    dmitensor.hpp
    effectivefield.hpp 
    effectivefield.cpp
    elasticforce.cu
    elasticforce.hpp
    elasticity.cu
    elasticity.hpp
    electricalpotential.hpp
    electricalpotential.cu
    energy.hpp 
    energy.cu
    exchange.hpp 
    exchange.cu 
    ferromagnet.cpp 
    ferromagnet.hpp 
    ferromagnetquantity.cpp 
    ferromagnetquantity.hpp
    magnet.cpp
    magnet.hpp 
    magnetoelasticfield.cu
    magnetoelasticfield.hpp
    magnetoelasticforce.cu
    magnetoelasticforce.hpp
    minimizer.cu
    minimizer.hpp
    mumaxworld.cpp
    mumaxworld.hpp
    neel.cu
    neel.hpp
    newell.cu
    newell.hpp
    poissonsystem.hpp
    poissonsystem.cu
    relaxer.hpp
    relaxer.cpp
    straintensor.cu
    straintensor.hpp
    strayfield.cpp
    strayfield.hpp
    strayfieldbrute.cu
    strayfieldbrute.hpp
    strayfieldfft.cu
    strayfieldfft.hpp
    strayfieldkernel.cu
    strayfieldkernel.hpp
    stt.cu
    stt.hpp
    thermalnoise.cu
    thermalnoise.hpp
    torque.cu 
    torque.hpp
    fullmag.cu
    fullmag.hpp 
    zeeman.cu
    zeeman.hpp
)

target_include_directories(physics PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(physics PUBLIC .)

target_link_libraries(physics PUBLIC core linsolver)

link_directories(${CUDA_LIBRARY_DIR})
target_link_libraries(physics PRIVATE cufft curand)
