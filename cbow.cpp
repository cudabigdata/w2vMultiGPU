#include "cbow.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>


extern std::vector<GPUTrainer> gpuTrainers;

extern int * table;
extern int vocab_size, layer1_size , layer1_size_aligned;
extern int negative , window;
extern int table_size;
// To batch data to minimize data transfer, sen stores words + alpha values
// alpha value start at offset = MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH



#define MAX_SOURCE_SIZE (0x100000)
#define openclCheck(err) { \
	if (err != CL_SUCCESS) { \
		printf("OpenCL error: %d: %s, line %d\n", err, __FILE__, __LINE__); \
		assert(err == CL_SUCCESS); \
	} \
}

GPUTrainer::GPUTrainer(cl_device_id device)
{
	int ret;
	device_id = device;
	// Create an OpenCL context
	context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	openclCheck(ret);
	// Create a command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	openclCheck(ret);

    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(ComputeUnits), &ComputeUnits, NULL); openclCheck(ret)
}

void GPUTrainer::initialWithSource(const char * source_str, size_t size){
	if (context != NULL){
		cl_int ret;
		program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
			(const size_t *)&size, &ret); openclCheck(ret);
		if (program == NULL)
		{
			printf("Failed to create CL program from source.\n");
			exit(0);
		}
		ret  = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (ret != CL_SUCCESS)
		{
			// Determine the reason for the error
			size_t len;
			ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
			char *buffer = (char *) calloc(len, sizeof(char));
			ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

			printf("Error in kernel:\n");
			printf("%s\n", buffer);
			free(buffer);
			clReleaseProgram(program);
			exit(0);
		}

		k_memset = clCreateKernel(program, "device_memset", &ret); openclCheck(ret) ;

//		size_t workgroup_size;
//		ret = clGetKernelWorkGroupInfo(k_memset, device_id, CL_KERNEL_WORK_GROUP_SIZE,
//		                                              sizeof(size_t), &workgroup_size, NULL);openclCheck(ret);
//		maxThreadsPerBlock = workgroup_size;

		ret = clGetKernelWorkGroupInfo(k_memset, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
	                                              sizeof(size_t), &wavefront_size, NULL); openclCheck(ret)

		if (wavefront_size == 32){
			k_cbow   = clCreateKernel(program, "device_cbow", &ret); openclCheck(ret) ;
		}
		else if (wavefront_size == 64){
			k_cbow   = clCreateKernel(program, "device_cbow64", &ret); openclCheck(ret) ;
		}else {
			printf("Unsupport wave front size of %d.\n", wavefront_size);
			assert(wavefront_size == 64);
		}
		real * h_expTable = (real *)malloc((EXP_TABLE_SIZE ) * sizeof(real));
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			h_expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
			h_expTable[i] = h_expTable[i] / (h_expTable[i] + 1);
		}

		d_expTable = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(real) * EXP_TABLE_SIZE, NULL, &ret); openclCheck(ret)
		ret = clEnqueueWriteBuffer(command_queue, d_expTable, CL_TRUE, 0,
				sizeof(real) * EXP_TABLE_SIZE, h_expTable, 0, NULL, NULL);openclCheck(ret)

		free(h_expTable);

		if (negative>0) {
			int syn1neg_size = vocab_size * layer1_size_aligned;
			d_syn1neg = clCreateBuffer(context, CL_MEM_READ_WRITE, syn1neg_size * sizeof(real), NULL, &ret);openclCheck(ret)

			// call memset kernel
			cl_int ret  = clSetKernelArg(k_memset, 0, sizeof(cl_mem), &d_syn1neg); openclCheck(ret);
			ret = clSetKernelArg(k_memset, 1, sizeof(syn1neg_size), &syn1neg_size);
			size_t global_size = syn1neg_size;
			ret =  clEnqueueNDRangeKernel(command_queue, k_memset, 1, NULL,&global_size, NULL, 0, NULL, NULL);
			openclCheck(ret);
			openclCheck(clFinish(command_queue));

			d_table = clCreateBuffer(context, CL_MEM_READ_ONLY, table_size * sizeof(int), NULL, &ret);openclCheck(ret)
			size_t table_mem = table_size * sizeof(int);

			ret= clEnqueueWriteBuffer(command_queue, d_table, CL_TRUE, 0,table_mem, table, 0, NULL, NULL);openclCheck(ret)
		}

		int syn0_size = vocab_size * layer1_size_aligned;
		d_syn0 = clCreateBuffer(context, CL_MEM_READ_WRITE, syn0_size * sizeof(real), NULL, &ret);openclCheck(ret)

		d_sen = clCreateBuffer(context, CL_MEM_READ_ONLY, (MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM) * sizeof(int), NULL, &ret);openclCheck(ret)

		d_random = clCreateBuffer(context, CL_MEM_READ_WRITE, MAX_SENTENCE_LENGTH * sizeof(unsigned int), NULL, &ret);openclCheck(ret)
		int h_random[MAX_SENTENCE_LENGTH];

		for (int i = 0 ; i < MAX_SENTENCE_LENGTH; i++) h_random[i] = (unsigned int) rand();
		ret = clEnqueueWriteBuffer(command_queue, d_random, CL_TRUE, 0, MAX_SENTENCE_LENGTH * sizeof(unsigned int), h_random, 0, NULL, NULL);openclCheck(ret)

		numBlock = MAX_SENTENCE_LENGTH / (BLOCK_SIZE/THREADS_PER_WORD) + 1;
		shared_mem_usage = (BLOCK_SIZE + (BLOCK_SIZE/THREADS_PER_WORD) * layer1_size_aligned * 2) * sizeof(real);

		this->setCbowArgs();
	}

	sen = (int*) malloc((MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM) * sizeof(int));
	posix_memalign((void **) &syn0, 128, (int) vocab_size * layer1_size_aligned * sizeof(real));

	bitmap.setSize(vocab_size);
}

void GPUTrainer::setCbowArgs(){
	cl_int ret;
	ret  = clSetKernelArg(k_cbow, 1, sizeof(layer1_size), &layer1_size); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 2, sizeof(layer1_size_aligned), &layer1_size_aligned); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 3, sizeof(window), &window); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 4, sizeof(negative), &negative); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 5, sizeof(table_size), &table_size); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 6, sizeof(vocab_size), &vocab_size); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 7, sizeof(d_sen), &d_sen); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 8, sizeof(d_table), &d_table); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 9, sizeof(d_syn0), &d_syn0); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 10, sizeof(d_syn1neg), &d_syn1neg); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 11, sizeof(d_random), &d_random); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 12, sizeof(d_expTable), &d_expTable); openclCheck(ret);
	ret  = clSetKernelArg(k_cbow, 13, shared_mem_usage , NULL); openclCheck(ret);
}

void GPUTrainer::cleanUpGPU(){

	if (d_syn1neg) openclCheck(clReleaseMemObject(d_syn1neg));
	if (d_syn0) openclCheck(clReleaseMemObject(d_syn0));
	if (d_sen) openclCheck(clReleaseMemObject(d_sen));
	if (d_random) openclCheck(clReleaseMemObject(d_random));
	if (d_table) openclCheck(clReleaseMemObject(d_table));

	if (sen) free(sen);
	if (syn0) free(syn0);
}


void initializeGPU()
{
	cl_uint platformCount;
	cl_platform_id* platforms;
	cl_uint deviceCount;
	cl_device_id* devices;
	char* value;
	size_t valueSize;
    cl_uint maxComputeUnits = 0;
    cl_int ret;
	ret = clGetPlatformIDs(0, NULL, &platformCount); openclCheck(ret);
	platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
	ret = clGetPlatformIDs(platformCount, platforms, NULL);  openclCheck(ret);
	printf("Detect %d platform available.\n",platformCount);
    for (unsigned int i= 0; i < platformCount; i++) {
        // get all devices
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);  openclCheck(ret)
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL); openclCheck(ret)
        printf("Platform %d. %d device available.\n", i+1, deviceCount );
        // for each device print critical attributes
        for (unsigned int j = 0; j < deviceCount; j++) {
            // print device name
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize); openclCheck(ret)
            value = (char*) malloc(valueSize);
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL); openclCheck(ret)
            printf("\t%d. Device: %s\n", j+1, value);
            free(value);

            // print hardware device version
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize); openclCheck(ret)
            value = (char*) malloc(valueSize);
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL); openclCheck(ret)
            printf("\t\t%d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            ret = clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize); openclCheck(ret)
            value = (char*) malloc(valueSize);
            ret = clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL); openclCheck(ret)
            printf("\t\t%d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            ret= clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize); openclCheck(ret)
            value = (char*) malloc(valueSize);
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL); openclCheck(ret)
            printf("\t\t%d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            cl_uint computeUnits;
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(computeUnits), &computeUnits, NULL); openclCheck(ret)
            printf("\t\t%d.%d Parallel compute units: %d\n\n", j+1, 4, computeUnits);
            maxComputeUnits += computeUnits;

            GPUTrainer newGPUTrainer(devices[j]);
            gpuTrainers.push_back(newGPUTrainer);

        }
        free(devices);
    }
//    printf("Select platform [1-%d]:", platformCount);
//    int selected_platform;
//    //scanf("%d", &selected_platform);
//    selected_platform = 1;
//
//    platform_id = platforms[selected_platform -1];
//    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
//    printf("Select device [1-%d]:", deviceCount);
//    int selected_device;
//   scanf("%d", &selected_device);
//    //selected_device = 1;
//    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
//    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
//    device_id = devices[selected_device -1];
//    free(devices);
//    free(platforms);

	char *source_str;
	FILE * fin = fopen("word2vec.cl", "r");
	if (fin == NULL)
	{
		printf( "Failed to opencl_kernel.cl file for reading:\n");
		exit(0);
	}
	source_str = (char*) malloc(MAX_SOURCE_SIZE);
	size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fin);
	fclose(fin);
	//printf("========SOURCE========\n %s", source_str);
	//printf("======================\n");

	// Build program for each GPU.
	for (unsigned int i = 0 ; i < gpuTrainers.size(); i++)
	{
		gpuTrainers[i].initialWithSource(source_str, source_size);
	}

	free(source_str);
	// Set working range for each GPUTrainer
	float start = 0;
	for (unsigned int i = 0 ; i < gpuTrainers.size(); i++)
	{
		int computeUnit = gpuTrainers[i].getComputeUnit();
		float end =(float)( computeUnit / (float) maxComputeUnits);
		gpuTrainers[i].setWorkingRange(start, start + end);
		start += end;
	}
}


void GPUTrainer::transferDataToGPU(){
	cl_int ret = clEnqueueWriteBuffer(command_queue, d_sen, CL_TRUE, 0,
			(MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM) * sizeof(int) , sen, 0, NULL, NULL);openclCheck(ret)
	openclCheck(clFinish(command_queue));
}

void GPUTrainer::getResultData(){
	cl_int ret = clEnqueueReadBuffer(command_queue, d_syn0, CL_TRUE, 0,
			 vocab_size * layer1_size_aligned * sizeof(real) , syn0, 0, NULL, NULL);openclCheck(ret)
	openclCheck(clFinish(command_queue));

}


void GPUTrainer::trainGPU(int sentence_num) {
	transferDataToGPU();
	cl_int ret  = clSetKernelArg(k_cbow, 0, sizeof(sentence_num), &sentence_num); openclCheck(ret);
	size_t global_workgroup = numBlock * BLOCK_SIZE;
	size_t local_workgroup = BLOCK_SIZE;

	ret =  clEnqueueNDRangeKernel(command_queue, k_cbow, 1, NULL,&global_workgroup, &local_workgroup, 0, NULL, NULL);
	openclCheck(ret);

}

void GPUTrainer::updateSyn0(float * g_syn0){
	cl_int ret = clEnqueueWriteBuffer(command_queue, d_syn0, CL_TRUE, 0,
			 vocab_size * layer1_size_aligned * sizeof(real) , g_syn0, 0, NULL, NULL);openclCheck(ret)
	openclCheck(clFinish(command_queue));
}

