CXXFLAGS =	 -g -Wall -fmessage-length=0

OBJS =		wiener2.o

LIBS =	`pkg-config --cflags --libs opencv`

TARGET =	wiener2

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
