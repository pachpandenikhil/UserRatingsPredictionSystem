uv_output.txt:
	This is the output for UV decomposition with command
		python uv.py mat.dat 5 5 2 10
	This program does not write the output to any files. It uses standard output to print them instead.

als_output.txt:
	This is the output file for ALS with command
		bin/spark-submit als.py mat.dat 5 5 4 10 5 als_output.txt
