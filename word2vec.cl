#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 102400
#define MAX_CODE_LENGTH 40
#define MAX_SENTENCE_NUM 6
#define ALIGNMENT_FACTOR 32
#define THREADS_PER_WORD 128
#define BLOCK_SIZE 128

kernel void device_memset(global float * array, int size){
	int idx = get_global_id(0);
	if (idx < size)
		array[idx] = 0;
}


void reduceInWarp(volatile local float * f, int idInWarp){

	for (unsigned int i=THREADS_PER_WORD /2; i>32; i>>=1) {
		if (idInWarp < i) {
			f[idInWarp] += f[idInWarp + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (idInWarp < 32){
		f[idInWarp] += f[idInWarp + 32];
		f[idInWarp] += f[idInWarp + 16];
		f[idInWarp] += f[idInWarp + 8];
		f[idInWarp] += f[idInWarp + 4];
		f[idInWarp] += f[idInWarp + 2];
		f[idInWarp] += f[idInWarp + 1];
	}
}

void reduceInWarp64(volatile local float * f, int idInWarp){

	for (unsigned int i=THREADS_PER_WORD /2; i>64; i>>=1) {
		if (idInWarp < i) {
			f[idInWarp] += f[idInWarp + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (idInWarp < 64){
		f[idInWarp] += f[idInWarp + 64];
		f[idInWarp] += f[idInWarp + 32];
		f[idInWarp] += f[idInWarp + 16];
		f[idInWarp] += f[idInWarp + 8];
		f[idInWarp] += f[idInWarp + 4];
		f[idInWarp] += f[idInWarp + 2];
		f[idInWarp] += f[idInWarp + 1];
	}
}

kernel void device_cbow(int sentence_num, int layer1_size, int layer1_size_aligned,
		int window, int negative, int table_size, int vocab_size,
		 global int * d_sen,  global int * d_table,
		 global float * d_syn0, global float *d_syn1neg,
		 global unsigned int * d_random,  global float * expTable, volatile local float * shared ){


	int sentence_position = (get_local_id(0) / THREADS_PER_WORD) + (get_local_size(0) / THREADS_PER_WORD) * get_group_id(0);
	int idInWarp = get_local_id(0) % THREADS_PER_WORD;


	volatile local float * f = &shared [ (get_local_id(0) / THREADS_PER_WORD) * THREADS_PER_WORD];
	volatile local float * neu1 = &shared [ BLOCK_SIZE + (get_local_id(0) / THREADS_PER_WORD) * layer1_size_aligned];
	volatile local float * neu1e= & shared[BLOCK_SIZE + (get_local_size(0) / THREADS_PER_WORD) * layer1_size_aligned + (get_local_id(0) / THREADS_PER_WORD) * layer1_size_aligned];

	if (sentence_position < MAX_SENTENCE_LENGTH) {
		unsigned int next_random = d_random[sentence_position];

		for (int sentence_idx = 0; sentence_idx < sentence_num; sentence_idx++){

			for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD) neu1[c] = 0;
			for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD) neu1e[c] = 0;



			next_random = next_random * (unsigned int) 1664525 + 1013904223;
			int b = next_random % window;
			int word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + sentence_position];
			// in -> hidden
			int cw = 0;
			for (int a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					int w = sentence_position - window + a;
					if (w < 0)
						continue;
					if (w>= MAX_SENTENCE_LENGTH)
						continue;
					int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];
					for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
						neu1[c] += d_syn0[c + last_word * layer1_size_aligned];

					cw++;
				}
			
			if (cw) {
				for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
					neu1[c] /= cw;
			
			// NEGATIVE SAMPLING
			int target, label;
			float alpha =((global float *) &d_sen[MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + sentence_idx])[0];

			if (negative > 0)

				for (int d = 0; d < negative + 1; d++) {


					if (d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned int) 1664525
								+ 1013904223;
						target = d_table[(next_random) % table_size];
						if (target == 0)
							target = next_random % (vocab_size - 1) + 1;
						if (target == word)
							continue;
						label = 0;
					}
					int l2 = target * layer1_size_aligned;
					f[idInWarp] = 0;
				
					
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD){
						f[idInWarp] += neu1[c] * d_syn1neg[c + l2];   
					}
					barrier(CLK_LOCAL_MEM_FENCE);
					// Do reduction here;
					for (unsigned int i=THREADS_PER_WORD /2; i>32; i>>=1) {
					if (idInWarp < i) {
						f[idInWarp] += f[idInWarp + i];
					}
					barrier(CLK_LOCAL_MEM_FENCE);
					}
					if (idInWarp < 32){
						f[idInWarp] += f[idInWarp + 32];
						f[idInWarp] += f[idInWarp + 16];
						f[idInWarp] += f[idInWarp + 8];
						f[idInWarp] += f[idInWarp + 4];
						f[idInWarp] += f[idInWarp + 2];
						f[idInWarp] += f[idInWarp + 1];
					}

					barrier(CLK_LOCAL_MEM_FENCE);
					
					float g;
					if (f[0] > MAX_EXP)
						g = (label - 1) * alpha;
					else if (f[0] < -MAX_EXP)
						g = (label - 0) * alpha;
					else
						g = (label - expTable[(int) ((f[0] + MAX_EXP)
									* (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

					//barrier(CLK_LOCAL_MEM_FENCE);	
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						neu1e[c] += g * d_syn1neg[c + l2];
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						d_syn1neg[c + l2] += g * neu1[c];
					
				}
			// hidden -> in
			for (int a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					int w = sentence_position - window + a;
					if (w < 0)
						continue;
					if (w >= MAX_SENTENCE_LENGTH)
						continue;
					int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];

					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						d_syn0[c + last_word * layer1_size_aligned] += neu1e[c];

				}
			}

		}// End for sentence_idx
		// Update d_random
		if (idInWarp == 0 ) d_random[sentence_position] = next_random;
	
	}
}


kernel void device_cbow64(int sentence_num, int layer1_size, int layer1_size_aligned,
		int window, int negative, int table_size, int vocab_size,
		global int * d_sen, global int * d_table,
		global float * d_syn0, global float *d_syn1neg,
		global unsigned int * d_random, __constant float * expTable, volatile local float * shared ){


	int sentence_position = (get_local_id(0) / THREADS_PER_WORD) + (get_local_size(0) / THREADS_PER_WORD) * get_group_id(0);
	int idInWarp = get_local_id(0) % THREADS_PER_WORD;


	volatile local float * f = shared + (get_local_id(0) / THREADS_PER_WORD) * THREADS_PER_WORD;
	volatile local float * neu1 = shared + BLOCK_SIZE + (get_local_id(0) / THREADS_PER_WORD) * layer1_size_aligned;
	volatile local float * neu1e= shared + BLOCK_SIZE + (get_local_size(0) / THREADS_PER_WORD) * layer1_size_aligned + (get_local_id(0) / THREADS_PER_WORD) * layer1_size_aligned;

	if (sentence_position < MAX_SENTENCE_LENGTH) {
		unsigned int next_random = d_random[sentence_position];

		for (int sentence_idx = 0; sentence_idx < sentence_num; sentence_idx++){

			for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD) neu1[c] = 0;
			for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD) neu1e[c] = 0;



			next_random = next_random * (unsigned int) 1664525 + 1013904223;
			int b = next_random % window;
			int word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + sentence_position];
			// in -> hidden
			int cw = 0;
			for (int a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					int w = sentence_position - window + a;
					if (w < 0)
						continue;
					if (w>= MAX_SENTENCE_LENGTH)
						continue;
					int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];
					for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
						neu1[c] += d_syn0[c + last_word * layer1_size_aligned];

					cw++;
				}
			
			if (cw) {
				for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
					neu1[c] /= cw;
			
			// NEGATIVE SAMPLING
			int target, label;
			float alpha =*((global float *) &d_sen[MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + sentence_idx]);

			if (negative > 0)

				for (int d = 0; d < negative + 1; d++) {


					if (d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned int) 1664525
								+ 1013904223;
						target = d_table[(next_random) % table_size];
						if (target == 0)
							target = next_random % (vocab_size - 1) + 1;
						if (target == word)
							continue;
						label = 0;
					}
					int l2 = target * layer1_size_aligned;
					f[idInWarp] = 0;
				
					
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD){
						f[idInWarp] += neu1[c] * d_syn1neg[c + l2];   
					}
					barrier(CLK_LOCAL_MEM_FENCE);
					// Do reduction here;
					reduceInWarp64(f, idInWarp);

					barrier(CLK_LOCAL_MEM_FENCE);
					
					float g;
					if (f[0] > MAX_EXP)
						g = (label - 1) * alpha;
					else if (f[0] < -MAX_EXP)
						g = (label - 0) * alpha;
					else
						g = (label - expTable[(int) ((f[0] + MAX_EXP)
									* (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

					//barrier(CLK_LOCAL_MEM_FENCE);	
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						neu1e[c] += g * d_syn1neg[c + l2];
					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						d_syn1neg[c + l2] += g * neu1[c];
					
				}
			// hidden -> in
			for (int a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					int w = sentence_position - window + a;
					if (w < 0)
						continue;
					if (w >= MAX_SENTENCE_LENGTH)
						continue;
					int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];

					for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
						d_syn0[c + last_word * layer1_size_aligned] += neu1e[c];

				}
			}
		}// End for sentence_idx
		// Update d_random
		if (idInWarp == 0 ) d_random[sentence_position] = next_random;
	}
}
