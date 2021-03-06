CC := g++ -std=c++11
CPPFLAGS := -Wall -O3
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

%.o: %.cpp
	@echo "------ Compile >> " $^ "to" $@ && \
	$(CC) $(CPPFLAGS) -c $^ -o $@ && \
	echo  "------ Success ------" || echo "------ Fail! ------"

clean:
	@$(RM) $(OBJS) && \
	echo "------ Clear ------"