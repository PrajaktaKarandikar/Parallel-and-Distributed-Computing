all: p1 p2

p1: p1.cu 
		nvcc  p1.cu -o p1
		
p2: p2.cu 
		nvcc  p2.cu -o p2
		

clean:
	    rm -f p1 p2