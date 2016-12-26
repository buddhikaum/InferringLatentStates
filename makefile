OBJS = Agent.o Variational.o main.o
CC = g++
CFLAGS = -Wall -Wshadow -std=c++11 -c -I/usr/local/include
LFLAGS = -L/usr/local/lib 

var-inf : $(OBJS) 
	$(CC) $(OBJS) $(LFLAGS) -o var-inf -larmadillo

Variational.o : Variational.h Variational.cpp Agent.h  
	$(CC) $(CFLAGS) Variational.cpp
	
Agent.o : Agent.h Agent.cpp 
	$(CC) $(CFLAGS) Agent.cpp


main.o : main.cpp Variational.h
	$(CC) $(CFLAGS) main.cpp


#CFLAGS = -Wall -Wshadow -std=c++11 -c -I/usr/local/include -I/usr/include/libxml2
