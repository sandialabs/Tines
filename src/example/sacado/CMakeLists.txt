# Empty list
SET(TINES_EXAMPLE_SOURCES "")

# Append examples that work for all device types
LIST(APPEND TINES_EXAMPLE_SOURCES
  Tines_View.cpp
  Tines_ToySacadoJacobian.cpp
  Tines_ToySacadoHessian.cpp
  Tines_ToySacadoStdVector.cpp
  Tines_ToySacadoReducer.cpp
  Tines_ToySacadoRayleighQuotient.cpp 
)

#
# Create examples for the above sources
#
FOREACH(TINES_EXAMPLE_SOURCE ${TINES_EXAMPLE_SOURCES} )
  STRING( REPLACE ".cpp" ".x" TINES_EXAMPLE_EXE ${TINES_EXAMPLE_SOURCE} )
  ADD_EXECUTABLE(${TINES_EXAMPLE_EXE} ${TINES_EXAMPLE_SOURCE})
  TARGET_LINK_LIBRARIES(${TINES_EXAMPLE_EXE} ${TINES_LINK_LIBRARIES})
  INSTALL(TARGETS ${TINES_EXAMPLE_EXE}
          PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
          DESTINATION "${CMAKE_INSTALL_PREFIX}/${TINES_INSTALL_EXAMPLE_PATH}")
ENDFOREACH()

