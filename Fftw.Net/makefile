ARCHIVE=fftw-3.3.8.tar.gz
SUBFOLDER=fftw-3.3.8
RUNTIME=linux-x64
PACKAGE=LibFFTW.NET
MAJOR=3
VERSION=3.3.8

./runtimes/linux-x64/native/libfftw$(MAJOR).so : ./$(SUBFOLDER)/.libs/libfftw$(MAJOR).so
	mkdir -p ./runtimes/$(RUNTIME)/native/ && cp ./$(SUBFOLDER)/.libs/libfftw$(MAJOR).so ./runtimes/$(RUNTIME)/native/

./$(SUBFOLDER)/.libs/libfftw$(MAJOR).so : ./$(SUBFOLDER)/configure
	cd ./$(SUBFOLDER)/ && ./configure --enable-threads --enable-shared && make

./$(SUBFOLDER)/configure :
	tar -xzvf ./$(ARCHIVE)

clean :
	rm -r ./$(SUBFOLDER)/
	rm -r ./runtimes/


