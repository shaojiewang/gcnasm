HIP_PATH?=$(wildcard /opt/rocm-3.1.0/hip)

ifeq (,$(HIP_PATH))
	$(error "pls choose a hip path")
endif

HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES=$(wildcard *.cc)
OBJECTS=$(patsubst %.cc,%.o,$(SOURCES))

EXECUTABLE=./algo_test

.PHONY: test

all: $(EXECUTABLE) test

CXXFLAGS=-g
CXX=$(HIPCC)

test: $(EXECUTABLE)
	echo $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@

.PHONY: clean
clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o