

PROJECT_NAME := Obj_Detection

JUMPSTART_BOARD := board_esp32_devkitc.h
EXTRA_COMPONENT_DIRS += $(PROJECT_PATH)/../components

include $(IDF_PATH)/make/project.mk
CPPFLAGS += -DJUMPSTART_BOARD=\"$(JUMPSTART_BOARD)\"
