file(READ "${INPUT}" DATA HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," DATA "${DATA}")
file(WRITE "${OUTPUT}" "static const unsigned char ${SYMBOL}[] = {${DATA}};\n")
file(APPEND "${OUTPUT}" "static const unsigned int ${SYMBOL}_len = sizeof(${SYMBOL});\n")
