default:
	python setup.py build_ext --inplace
clean:
	rm ./vectorize.so
