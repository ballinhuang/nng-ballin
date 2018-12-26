CC := nvcc -std=c++11
CPPFLAGS := -O3 -D_FORCE_INLINES
SRCS := $(shell ls *.cpp)
OBJS:=$(subst .cpp,.o,$(SRCS))
RM:=rm -f

all: 
	$(MAKE) nng
	$(MAKE) clean

nng: $(OBJS)
	@echo "------ Linking ------" && \
	$(CC) $(CPPFLAGS)  $^ -o $@ && \
	echo  "------ Success ------" || echo "------ Fail! ------"

%.o: %.c*
	@echo "------ Compile >> " $^ "to" $@ && \
	$(CC) $(CPPFLAGS) --device-c --x=cu $^ && \
	echo  "------ Success ------" || echo "------ Fail! ------"


clean:
	@$(RM) $(OBJS) && \
	echo "------ Clear ------"