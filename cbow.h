/*
 * cbow.h
 *
 *  Created on: Aug 29, 2015
 *      Author: gpgpu
 */

#ifndef CBOW_H_
#define CBOW_H_

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 102400
#define MAX_CODE_LENGTH 40
#define MAX_SENTENCE_NUM 6
#define ALIGNMENT_FACTOR 32
#define THREADS_PER_WORD 128
#define BLOCK_SIZE 128
#define MAX_GPU_SUPPORT 8
typedef float real;

#define NUM_ITERATION_DO_SYNC_SYN0 5

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class MyBitMap{
private:
	unsigned int * bits;
	unsigned int size;
public:
	MyBitMap(){
		bits = NULL;
		size = 0;
	}
	void setSize(unsigned int num){
		size = (num + 31) / 32;
		bits = new unsigned int[size];
		for (unsigned int i = 0; i < size; i++)
			bits[i] = 0;
	}
	~MyBitMap()
	{
		delete[] bits;
	}
	void setBit(unsigned int index){
		unsigned int i = index / 32;
		unsigned int flag = 1;
		flag = flag << (31 - index % 32);
		bits[i] = bits[i] | flag;
	}
	unsigned int getBit(unsigned int index){
		unsigned int i = index / 32;
		unsigned int flag = 1;
		flag = flag << (31 - index % 32);
		 if ((flag & bits[i]) == 0)
			 return 0;
		 else
			 return 1;
	}
};

class GPUTrainer
{
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_device_id device_id;
	cl_kernel k_memset;
	cl_kernel k_cbow;
	unsigned int wavefront_size;
	cl_platform_id platform_id;

	//
	cl_mem d_syn0;
	cl_mem d_syn1neg;
	cl_mem d_sen;
	cl_mem d_random;
	cl_mem d_table;
	cl_mem d_expTable;

	int numBlock;
	int shared_mem_usage;

	int * sen;
	float * syn0;
	float * syn1neg;
	int ComputeUnits;
	float startOffset;
	float endOffset;
	void setCbowArgs();
	void transferDataToGPU();

public:
	MyBitMap bitmap;
	int getComputeUnit() {  return ComputeUnits;}
	int * getSentencePtr() { return sen;}
	GPUTrainer(cl_device_id device);
	void initialWithSource(const char * src, size_t size);
	void cleanUpGPU();
	void trainGPU(int sentence_num);
	void getResultData();
	void updateSyn0(float * g_syn0);
	float * getSyn0() { return syn0;}
	void updateSyn1Neg(float * g_syn1neg);
	float * getSyn1Neg() { return syn1neg;}
	void setWorkingRange(float start, float end) { startOffset = start; endOffset = end;}
	float getStart() { return startOffset;}
	float getEnd() { return endOffset;}
};


void initializeGPU();


#endif /* CBOW_H_ */
