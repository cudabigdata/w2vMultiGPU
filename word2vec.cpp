//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include "cbow.h"

std::vector<GPUTrainer> gpuTrainers;

const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

// Precision of float numbers

struct vocab_word {
	int cn;
	int *point;
	char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5,
		num_threads = 12, min_reduce = 1;
int *vocab_hash;
int vocab_max_size = 1000, vocab_size = 0, layer1_size = 100,
		layer1_size_aligned;
;
unsigned int train_words = 0, iter = 5;
int file_size = 0, classes = 0;
unsigned int word_count_actual = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0;

clock_t start;

int benchmark = 0;
int hs = 0, negative = 5;
int table_size = 1e8;
int *table;


#define IO_BLOCK_SIZE  4096
int buf_io_fd[MAX_GPU_SUPPORT] = { -1, -1, -1, -1, -1, -1, -1, -1 };
int end_flag[MAX_GPU_SUPPORT] = { 0, 0, 0, 0, 0, 0, 0, 0 };
char* buf[MAX_GPU_SUPPORT] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
char word[MAX_GPU_SUPPORT][MAX_STRING];

ssize_t cur_pos[MAX_GPU_SUPPORT] = { 0, 0, 0, 0, 0, 0, 0, 0 };
ssize_t cur_end[MAX_GPU_SUPPORT] = { 0, 0, 0, 0, 0, 0, 0, 0 };

int fill_buffer(int id) {
	ssize_t rd = 0;

	while ((rd = read(buf_io_fd[id], &(buf[id][cur_pos[id]]),
			IO_BLOCK_SIZE - cur_pos[id])) < 0) {
		if (errno == EINTR)
			continue;

		// all the other errors read can return should never happen, if they do
		// something has gone very wrong, and there's nothing we can really do about it.
		perror("read");
		abort();
	}

	cur_end[id] = cur_pos[id] + rd;

	if (rd != IO_BLOCK_SIZE - cur_pos[id]) {
		buf[id][cur_pos[id] + rd] = '\0';
	}
	return 0;
}

#define isdelim(c) (((c) == ' ')  | ((c) == '\t') | ((c) == '\n') | ((c) == '\r'))

void buffered_readWord(int id) {

	if (cur_pos[id] >= IO_BLOCK_SIZE) {
		cur_pos[id] = 0;
		fill_buffer(id);
	} else if (cur_pos[id] >= cur_end[id]) { // EOF
		end_flag[id] = 1;
		return;
	}

	// look for start of token (first non-whitespace character)
	while (isdelim(buf[id][cur_pos[id]]) && buf[id][cur_pos[id]]) {
		cur_pos[id]++;
		if (cur_pos[id] >= IO_BLOCK_SIZE) {
			cur_pos[id] = 0;
			fill_buffer(id);
		}
	}

	ssize_t ptmp = cur_pos[id]; // need to rember start of token

	while (!isdelim(buf[id][cur_pos[id]]) && buf[id][cur_pos[id]]) // scan for end of token
	{
		cur_pos[id]++;
		if (cur_pos[id] >= IO_BLOCK_SIZE) {
			// copy already looked at part to begining
			// should never overlap, at least for tokens shorter than IO_BLOCK_SIZE/2
			memcpy(buf[id], &(buf[id][ptmp]),
					sizeof(char) * (IO_BLOCK_SIZE - ptmp));

			cur_pos[id] = IO_BLOCK_SIZE - ptmp;
			ptmp = 0;
			fill_buffer(id);
		}
	}

	ssize_t wordlen = (cur_pos[id] - ptmp);
	if (wordlen >= MAX_STRING)
		wordlen = MAX_STRING - 1;

	buf[id][cur_pos[id]] = '\0'; // replace space with null
	memcpy(word[id], &(buf[id][ptmp]), wordlen);
	word[id][wordlen] = '\0';
	cur_pos[id]++;
}

void reset_read_word(int id) {
	file_size = lseek(buf_io_fd[id], 0, SEEK_END);
	off_t ret = lseek(buf_io_fd[id], (file_size / num_threads) * id, SEEK_SET);

	if (ret < 0) {
		perror("lseek");
	}
	end_flag[id] = 0;
	cur_pos[id] = 0;
	cur_end[id] = 0;
	//printf("calling fill from reset\n");
	fill_buffer(id);
}

int open_buffered_file(int id) {
	if (buf_io_fd[id] != -1) {
		printf("file already open");
		exit(1);
	}
	buf_io_fd[id] = open(train_file, O_RDONLY);
	if (buf_io_fd[id] == -1)
		perror("open");

	size_t alignment = 4096;
	if (posix_memalign((void**) &buf[id], alignment, IO_BLOCK_SIZE) != 0) {
		perror("posix_memalign");
	}
	reset_read_word(id);
	return 1;

}

int close_buffered_file(int id) {
	if (buf_io_fd[id] == -1) {
		printf("file already closed");
		exit(1);
	}

	close(buf_io_fd[id]);
	buf_io_fd[id] = -1;
	free(buf[id]);
	return 1;
}

void InitUnigramTable() {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;
	table = (int *) malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++)
		train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double) table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / train_words_pow;
		}
		if (i >= vocab_size)
			i = vocab_size - 1;
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13)
			continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				//     if (ch == '\n') ungetc(ch, fin);
				break;
			}
			//   if (ch == '\n') {
			//     strcpy(word, (char *)"</s>");
			//    return;
			//  } else continue;
			continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1)
			a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned int a, hash = 0;
	for (a = 0; a < strlen(word); a++)
		hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1)
			return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word))
			return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(int id) {
	//char word[MAX_STRING];
	buffered_readWord(id);
	if (end_flag[id])
		return -1;
	return SearchVocab(word[id]);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING)
		length = MAX_STRING;
	vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *) realloc(vocab,
				vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1)
		hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if ((vocab[a].cn < min_count) && (a != 0)) {
			vocab_size--;
			free(vocab[a].word);
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *) realloc(vocab,
			(vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	/*for (a = 0; a < vocab_size; a++) {
	 vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
	 vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	 }*/
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++)
		if (vocab[a].cn > min_reduce) {
			vocab[b].cn = vocab[a].cn;
			vocab[b].word = vocab[a].word;
			b++;
		} else
			free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}
/*
 // Create binary Huffman tree using the word counts
 // Frequent words will have short uniqe binary codes
 void CreateBinaryTree() {
 int a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
 char code[MAX_CODE_LENGTH];
 int *count = (int *)calloc(vocab_size * 2 + 1, sizeof(int));
 int *binary = (int *)calloc(vocab_size * 2 + 1, sizeof(int));
 int *parent_node = (int *)calloc(vocab_size * 2 + 1, sizeof(int));
 for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
 for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e9;
 pos1 = vocab_size - 1;
 pos2 = vocab_size;
 // Following algorithm constructs the Huffman tree by adding one node at a time
 for (a = 0; a < vocab_size - 1; a++) {
 // First, find two smallest nodes 'min1, min2'
 if (pos1 >= 0) {
 if (count[pos1] < count[pos2]) {
 min1i = pos1;
 pos1--;
 } else {
 min1i = pos2;
 pos2++;
 }
 } else {
 min1i = pos2;
 pos2++;
 }
 if (pos1 >= 0) {
 if (count[pos1] < count[pos2]) {
 min2i = pos1;
 pos1--;
 } else {
 min2i = pos2;
 pos2++;
 }
 } else {
 min2i = pos2;
 pos2++;
 }
 count[vocab_size + a] = count[min1i] + count[min2i];
 parent_node[min1i] = vocab_size + a;
 parent_node[min2i] = vocab_size + a;
 binary[min2i] = 1;
 }
 // Now assign binary code to each vocabulary word
 for (a = 0; a < vocab_size; a++) {
 b = a;
 i = 0;
 while (1) {
 code[i] = binary[b];
 point[i] = b;
 i++;
 b = parent_node[b];
 if (b == vocab_size * 2 - 2) break;
 }
 vocab[a].codelen = i;
 vocab[a].point[0] = vocab_size - 2;
 for (b = 0; b < i; b++) {
 vocab[a].code[i - b - 1] = code[b];
 vocab[a].point[i - b] = point[b] - vocab_size;
 }
 }
 free(count);
 free(binary);
 free(parent_node);
 }
 */
void LearnVocabFromTrainFile() {
	//char word[MAX_STRING];
	//FILE *fin;
	int a, i;
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	/*fin = fopen(train_file, "rb");
	 if (fin == NULL) {
	 printf("ERROR: training data file not found!\n");
	 exit(1);
	 }*/
	open_buffered_file(0);
	vocab_size = 0;
	AddWordToVocab((char *) "</s>");
	while (1) {
		buffered_readWord(0);
		if (end_flag[0])
			break;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%dK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word[0]);
		if (i == -1) {
			a = AddWordToVocab(word[0]);
			vocab[a].cn = 1;
		} else
			vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7)
			ReduceVocab();
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %d\n", vocab_size);
		printf("Words in train file: %d\n", train_words);
	}
	//file_size = ftell(fin);
	close_buffered_file(0);
}

void SaveVocab() {
	int i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++)
		fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}
/*
 void ReadVocab() {
 int a, i = 0;
 char c;
 char word[MAX_STRING];
 FILE *fin = fopen(read_vocab_file, "rb");
 if (fin == NULL) {
 printf("Vocabulary file not found\n");
 exit(1);
 }
 for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
 vocab_size = 0;
 while (1) {
 ReadWord(word, fin);
 if (feof(fin)) break;
 a = AddWordToVocab(word);
 fscanf(fin, "%d%c", &vocab[a].cn, &c);
 i++;
 }
 SortVocab();
 if (debug_mode > 0) {
 printf("Vocab size: %d\n", vocab_size);
 printf("Words in train file: %d\n", train_words);
 }
 fin = fopen(train_file, "rb");
 if (fin == NULL) {
 printf("ERROR: training data file not found!\n");
 exit(1);
 }
 fseek(fin, 0, SEEK_END);
 file_size = ftell(fin);
 fclose(fin);
 }
 */
void InitNet() {
	int a, b;
	unsigned int next_random = 1;
	a = posix_memalign((void **) &syn0, 128,
			(int) vocab_size * layer1_size_aligned * sizeof(real));
	if (syn0 == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	for (a = 0; a < vocab_size; a++)
		for (b = 0; b < layer1_size; b++) {
			next_random = next_random * (unsigned int) 1664525 + 1013904223;
			syn0[a * layer1_size_aligned + b] = (((next_random & 0xFFFF)
					/ (real) 65536) - 0.5) / layer1_size;
		}
	//CreateBinaryTree();
}

void *TrainModelThread(void *id) {
	int word, sentence_length = 0;
	unsigned int word_count = 0, last_word_count = 0;
	unsigned int next_random = (long) id;
	int fid = (int) (long) id;
	int sentence_num;
	clock_t now;
	int * sen = gpuTrainers[fid].getSentencePtr();
	real * alpha_ptr = (float *) sen + MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH;
	//FILE *fi = fopen(train_file, "rb");
	//fseek(fi, file_size / (int)num_threads * (long)id, SEEK_SET);
	//printf("opening file\n");

	open_buffered_file(fid);
	//printf("resetting file\n");
	//reset_read_word();

	sentence_length = 0;
	sentence_num = 0;
	int count_kernels = 0;
	while (1) {
		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now = clock();
				printf(
						"%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ",
						13, alpha,
						word_count_actual / (real) (iter * train_words + 1)
								* 100,
						word_count_actual
								/ ((real) (now - start + 1)
										/ (real) CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			alpha = starting_alpha
					* (1 - word_count_actual / (real) (iter * train_words + 1));
			if (alpha < starting_alpha * 0.0001)
				alpha = starting_alpha * 0.0001;
		}

		while (1) {
			word = ReadWordIndex(fid);
			if (end_flag[fid])
				break;
			if (word == -1)
				continue;
			word_count++;
			if (word == 0)
				break;
			// The subsampling randomly discards frequent words while keeping the ranking same
			if (sample > 0) {
				real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1)
						* (sample * train_words) / vocab[word].cn;
				next_random = next_random * (unsigned int) 1664525 + 1013904223;
				if (ran < (next_random & 0xFFFF) / (real) 65536)
					continue;
			}
			sen[sentence_num * MAX_SENTENCE_LENGTH + sentence_length] = word;
			sentence_length++;
			if (sentence_length >= MAX_SENTENCE_LENGTH) {
				alpha_ptr[sentence_num] = alpha;
				sentence_num++;
				sentence_length = 0;
				if (sentence_num >= MAX_SENTENCE_NUM)
					break;
			}
		}

		if (benchmark > 0 && count_kernels == benchmark)
			exit(1);
		// Do GPU training here
		gpuTrainers[fid].trainGPU(sentence_num);
		count_kernels++;
		//////////////////////
		sentence_num = 0;
		sentence_length = 0;

		if (end_flag[fid] || (word_count > train_words / num_threads)) {
			word_count_actual += word_count - last_word_count;
			break;
		}

	}

	gpuTrainers[fid].getResultData();
	close_buffered_file(fid);
	pthread_exit(NULL);
}

void TrainModel() {
	long a, b;
	FILE *fo;

	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	LearnVocabFromTrainFile();
	/*if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	 if (save_vocab_file[0] != 0) SaveVocab();*/
	if (output_file[0] == 0)
		return;
	InitNet();
	if (negative > 0)
		InitUnigramTable();
	initializeGPU();
	num_threads = gpuTrainers.size();
	pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
	start = clock();
	// loop iteration
	for (unsigned int local_iter = 0; local_iter < iter; local_iter++){
		// distribute global syn0 to all GPUTrainer's syn0
		for (int i = 0; i < num_threads; i++)
			gpuTrainers[i].updateSyn0(syn0);
		// launch threads
		for (a = 0; a < num_threads; a++)
			pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
		for (a = 0; a < num_threads; a++)
			pthread_join(pt[a], NULL);
		// update global syn0 from all GPUTrainer's syn0
		for (a = 0; a < vocab_size ; a++)
			for (b = 0; b < layer1_size; b++)
			{
				float value = 0;
				int index = a * layer1_size_aligned + b;
				for (int i = 0 ; i < num_threads; i++)
					value += gpuTrainers[i].getSyn0()[index];
				syn0[index] = value / num_threads;
			}
	}


//	cleanUpGPU();
	fo = fopen(output_file, "wb");
	if (classes == 0) {
		// Save the word vectors
		fprintf(fo, "%d %d\n", vocab_size, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			fprintf(fo, "%s ", vocab[a].word);
			if (binary)
				for (b = 0; b < layer1_size; b++)
					fwrite(&syn0[a * layer1_size_aligned + b], sizeof(real), 1,
							fo);
			else
				for (b = 0; b < layer1_size; b++)
					fprintf(fo, "%f ", syn0[a * layer1_size_aligned + b]);
			fprintf(fo, "\n");
		}
	}
	fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf(
				"\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf(
				"\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf(
				"\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
		printf("\t-negative <int>\n");
		printf(
				"\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
//		printf("\t-threads <int>\n");
//		printf("\t\tUse <int> threads (default 12)\n");
		printf("\t-iter <int>\n");
		printf("\t\tRun more training iterations (default 5)\n");
		printf("\t-min-count <int>\n");
		printf(
				"\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf(
				"\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		printf("\t-classes <int>\n");
		printf(
				"\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf(
				"\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf(
				"\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf(
				"\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-cbow <int>\n");
		printf(
				"\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
		printf("\nExamples:\n");
		printf(
				"./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
		return 0;
	}
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	if ((i = ArgPos((char *) "-size", argc, argv)) > 0) {
		layer1_size = atoi(argv[i + 1]);
		layer1_size_aligned = ((layer1_size - 1) / ALIGNMENT_FACTOR + 1)
				* ALIGNMENT_FACTOR;
	}
	if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
		strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
		strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
		strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
		debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)
		binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0)
		cbow = atoi(argv[i + 1]);
	if (cbow)
		alpha = 0.05;
	if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
		alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-output", argc, argv)) > 0)
		strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
		window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
		sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-hs", argc, argv)) > 0)
		hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
		negative = atoi(argv[i + 1]);
//	if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)
//		num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-iter", argc, argv)) > 0)
		iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
		min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-classes", argc, argv)) > 0)
		classes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-benchmark", argc, argv)) > 0)
		benchmark = atoi(argv[i + 1]);
	vocab = (struct vocab_word *) calloc(vocab_max_size,
			sizeof(struct vocab_word));
	vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));

	TrainModel();
	return 0;
}
