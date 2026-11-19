/********************************************************************************/
/* @ File: MOC_AllocAndFreeMem.c
/*   introduction: functions of allocating memory for pointers
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_CommonMacro.h"
#include "MOC_NTE_Solver.h"
#include "MOC_Output.h"
#include "MOC_Track.h"

void CheckDataTypeRange(int _procID, FILE *_LogFile)
{
	M_UINT64 _size=0;
	int _width1=20,_width2=2;

	if (_procID==0) {
	printf("\n [  INFO   ]  Check the data size on current hardware");
	printf("\n              %-*s%-*s",_width1,"DataType",_width2,"Size (Byte)");

	_size = (M_UINT64)sizeof(char);
	printf("\n              %-*s%-*llu",_width1,"char",_width2,_size);
	_size = (M_UINT64)sizeof(signed char);
	printf("\n              %-*s%-*llu",_width1,"signed char",_width2,_size);
	_size = (M_UINT64)sizeof(unsigned char);
	printf("\n              %-*s%-*llu",_width1,"unsigned char",_width2,_size);
	_size = (M_UINT64)sizeof(short);
	printf("\n              %-*s%-*llu",_width1,"short",_width2,_size);
	_size = (M_UINT64)sizeof(unsigned short);
	printf("\n              %-*s%-*llu",_width1,"unsigned short",_width2,_size);
	_size = (M_UINT64)sizeof(int);
	printf("\n              %-*s%-*llu",_width1,"int",_width2,_size);
	_size = (M_UINT64)sizeof(unsigned int);
	printf("\n              %-*s%-*llu",_width1,"unsigned int",_width2,_size);
	_size = (M_UINT64)sizeof(long);
	printf("\n              %-*s%-*llu",_width1,"long",_width2,_size);
	_size = (M_UINT64)sizeof(unsigned long);
	printf("\n              %-*s%-*llu",_width1,"unsigned long",_width2,_size);
	_size = (M_UINT64)sizeof(long long);
	printf("\n              %-*s%-*llu",_width1,"long",_width2,_size);
	_size = (M_UINT64)sizeof(unsigned long long);
	printf("\n              %-*s%-*llu",_width1,"unsigned long long",_width2,_size);
	_size = (M_UINT64)sizeof(size_t);
	printf("\n              %-*s%-*llu",_width1,"size_t",_width2,_size);
	_size = (M_UINT64)sizeof(float);
	printf("\n              %-*s%-*llu",_width1,"float",_width2,_size);
	_size = (M_UINT64)sizeof(double);
	printf("\n              %-*s%-*llu",_width1,"double",_width2,_size);
	_size = (M_UINT64)sizeof(long double);
	printf("\n              %-*s%-*llu",_width1,"long double",_width2,_size);
	}



}

/**
* @ Function: AllocMem_i32_i32()
*   introduction: allocate memory for int *pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _parameter - parameter name
*   Return: int *ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
M_INT32 *AllocMem_i32_i32(M_INT32 _dim, char *_parameter)
{
	M_UINT64 _dataNumber = 0, _dim1;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	M_INT32 *_pointer = NULL;

	_dim1 = (size_t)(_dim);
	if ((_pointer = (M_INT32 *)calloc(_dim1, sizeof(M_INT32))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_i32_i32\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_i32_i32\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer));
	_dataNumber += (M_UINT64)(_dim1);
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(M_INT32));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "M_INT32*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

M_INT32 *AllocMem_i32_I(size_t _dim1, char *_parameter)
{
	M_UINT64 _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	M_INT32 *_pointer = NULL;

	if ((_pointer = (int *)calloc(_dim1, sizeof(M_INT32))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_i32_i32\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_i32_i32\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer));
	_dataNumber += (M_UINT64)(_dim1);
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(M_INT32));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "M_INT32*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}


M_INT32 *FreeMem_i32_I(M_INT32 *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	if (_pointer != NULL) {
		free(_pointer);
	}
	_pointer = NULL;
	return _pointer;
}

M_UINT64 *AllocMem_ui64_I(size_t _dim1, char *_parameter)
{
	M_UINT64 _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	M_UINT64 *_pointer = NULL;
	if ((_pointer = (M_UINT64 *)calloc(_dim1, sizeof(M_UINT64))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_ui64_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_ui64_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer));
	_dataNumber += (M_UINT64)(_dim1);
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(M_UINT64));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "M_UINT64*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

M_UINT64 *FreeMem_ui64_I(M_UINT64 *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	if (_pointer != NULL) {
		free(_pointer);
	}
	_pointer = NULL;
	return _pointer;
}

M_INT32 *AllocMem_int_I(int _dim1, char *_parameter)
{
	M_UINT64 _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	M_INT32 *_pointer = NULL;

	if ((_pointer = (int *)calloc(_dim1, sizeof(M_INT32))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_i32_i32\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_i32_i32\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer));
	_dataNumber += (M_UINT64)(_dim1);
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(M_INT32));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "M_INT32*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

M_INT32 *FreeMem_int_I(M_INT32 *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	if (_pointer != NULL) {
		free(_pointer);
	}
	_pointer = NULL;
	return _pointer;
}

FP_PRECISION *AllocMem_FP_I(size_t _dim1, char *_parameter)
{
	M_UINT64 _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION *_pointer = NULL;
	
	if ((_pointer = (FP_PRECISION *)calloc(_dim1, sizeof(FP_PRECISION))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer));
	_dataNumber += (M_UINT64)(_dim1);

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

/**
* @ Function: FreeMem_i32_I()
*   introduction: free memory for int *pointer
* @ Parameter List: _pointer - memory pointer, need to be freed
*                   _parameter - parameter name
*   Return: int *ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/

/**
* @ Function: AllocMem_int_II()
*   introduction: allocate memory for int **pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _dim2 - size of dimension 2
*                   _parameter - parameter name
*   Return: int **ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int **AllocMem_int_II(int _dim1, int _dim2, char *_parameter)
{
	int _iter1;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	int **_pointer = NULL;
	if ((_pointer = (int **)calloc(_dim1, sizeof(int *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_II\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_II\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (int *)calloc(_dim2, sizeof(int))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_II\" - %s[%d]\n", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_II\" - local variable [%d]\n", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber += _dim2;
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(int));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "int**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}

/**
* @ Function: FreeMem_int_II()
*   introduction: free memory for int *pointer
* @ Parameter List: _pointer - memory pointer, need to be freed
*                   _dim1 - size of dimension 1
*                   _parameter - parameter name
*   Return: int *ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int **FreeMem_int_II(int **_pointer, int _dim1, char *_parameter)
{
	int _iter1;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}

	for (_iter1 = 0; _iter1<_dim1; _iter1++){
        if (_pointer[_iter1] != NULL){
            free(_pointer[_iter1]);
            _pointer[_iter1] = NULL;
        }
    }

	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

/**
* @ Function: AllocMem_int_IP()
*   introduction: allocate memory for int **pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _dim2 - size of dimension 2
*                   _parameter - parameter name
*   Return: int **ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int **AllocMem_int_IP(int _dim1, int *_dim2, char *_parameter)
{
	int _iter1;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	int **_pointer = NULL;
	if ((_pointer = (int **)calloc(_dim1, sizeof(int *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IP\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IP\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (int *)calloc(_dim2[_iter1], sizeof(int))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IP\" - %s[%d]\n", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IP\" - local variable [%d]\n", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber += _dim2[_iter1];
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(int));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "int**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}
/**
* @ Function: AllocMem_int_IPplusI()
*   introduction: allocate memory for int **pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _dim2 - size of dimension 2
*                   _plus - number need to be plused on _dim2[]
*                   _parameter - parameter name
*   Return: int **ponter
* @ Created     : 2018/05/20, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int **AllocMem_int_IPplusI(int _dim1, int *_dim2,int _plus, char *_parameter)
{
	int _iter1;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	int **_pointer = NULL;
	if ((_pointer = (int **)calloc(_dim1, sizeof(int *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IPplusI\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IPplusI\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (int *)calloc(_dim2[_iter1]+_plus, sizeof(int))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IPplusI\" - %s[%d]\n", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_IPplusI\" - local variable [%d]\n", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber += _dim2[_iter1]+_plus;
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(int));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "int**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}
/**
* @ Function: FreeMem_int_IP()
*   introduction: free memory for int *pointer
* @ Parameter List: _pointer - memory pointer, need to be freed
*                   _dim1 - size of dimension 1
*                   _parameter - parameter name
*   Return: int **ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int **FreeMem_int_IP(int **_pointer, int _dim1, char *_parameter)
{
	int _iter1;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
        if (_pointer[_iter1] != NULL) {
            free(_pointer[_iter1]);
            _pointer[_iter1] = NULL;
        }
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

/**
* @ Function: AllocMem_int_III()
*   introduction: allocate memory for int **pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _dim2 - size of dimension 2
*                   _dim3 - size of dimension 3
*                   _parameter - parameter name
*   Return: int ***ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int ***AllocMem_int_III(int _dim1, int _dim2, int _dim3, char *_parameter)
{
	int _iter1, _iter2;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	int ***_pointer = NULL;
	if ((_pointer = (int ***)calloc(_dim1, sizeof(int **))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_III\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_III\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (int **)calloc(_dim2, sizeof(int *))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_III\" - %s[%d]\n", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_III\" - local variable [%d]\n", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer[_iter1]));
		for (_iter2 = 0; _iter2<_dim2; _iter2++) {
			if ((_pointer[_iter1][_iter2] = (int *)calloc(_dim3, sizeof(int))) == NULL) {
				if (strcmp(_parameter,CHAR_NONE) != 0) {
					printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_III\" - %s[%d][%d]\n", _parameter, _iter1, _iter2);
				}
				else {
					printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_int_III\" - local variable [%d][%d]\n", _iter1, _iter2);
				}
				fflush(stdout);
				exit(69);
			}
			_pointerSizeMB += (FP_PRECISION)(sizeof(_pointer[_iter1][_iter2]));
			_dataNumber = _dataNumber + _dim3;
		}
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(int));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "int***", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}

/**
* @ Function: FreeMem_int_III()
*   introduction: free memory for int *pointer
* @ Parameter List: _pointer - memory pointer, need to be freed
*                   _dim1 - size of dimension 1
*                   _dim2 - size of dimension 2
*                   _parameter - parameter name
*   Return: int **ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int ***FreeMem_int_III(int ***_pointer, int _dim1, int _dim2, char *_parameter)
{
	int _iter1, _iter2;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		for (_iter2 = 0; _iter2<_dim2; _iter2++) {
            if (_pointer[_iter1][_iter2] != NULL) {
                free(_pointer[_iter1][_iter2]);
                _pointer[_iter1][_iter2] = NULL;
            }
		}
        if (_pointer[_iter1] != NULL) {
            free(_pointer[_iter1]);
            _pointer[_iter1] = NULL;
        }
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

/**
* @ Function: AllocMem_double_II2P()
*   introduction: allocate memory for int **pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _dim2 - size of dimension 2
*                   _dim3 - size of dimension 3
*                   _parameter - parameter name
*   Return: int ***ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: FreeMem_double_II2P()
*   introduction: free memory for int *pointer
* @ Parameter List: _pointer - memory pointer, need to be freed
*                   _dim1 - size of dimension 1
*                   _dim2 - size of dimension 2
*                   _parameter - parameter name
*   Return: int **ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int ***FreeMem_double_II2P(int ***_pointer, int _dim1, int _dim2, char *_parameter)
{
	int _iter1, _iter2;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		for (_iter2 = 0; _iter2<_dim2; _iter2++) {
			free(_pointer[_iter1][_iter2]);
		}
		free(_pointer[_iter1]);
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

/**
* @ Function: AllocMem_FP_I_1()
*   introduction: allocate memory for FP_PRECISION *pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _parameter - parameter name
*   Return: FP_PRECISION *ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
FP_PRECISION *AllocMem_FP_I_1(int _dim1, char *_parameter)
{
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION *_pointer = NULL;
	if ((_pointer = (FP_PRECISION *)calloc(_dim1, sizeof(FP_PRECISION))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_I_1\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_I_1\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer));
	_dataNumber = _dataNumber + _dim1;

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}


FP_PRECISION_GEO *AllocMem_FP_I_Geo(int _dim1, char *_parameter)
{
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION_GEO *_pointer = NULL;
	if ((_pointer = (FP_PRECISION_GEO *)calloc(_dim1, sizeof(FP_PRECISION_GEO))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_I_1\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_I_1\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer));
	_dataNumber = _dataNumber + _dim1;

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION_GEO));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION_GEO*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

FP_PRECISION_GEO *FreeMem_FP_I_Geo(FP_PRECISION_GEO *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	if (_pointer != NULL) {
		free(_pointer);
	}
	
	_pointer = NULL;
	return _pointer;
}


FP_PRECISION_FLUX *AllocMem_FP_Flux(size_t _dim1, char *_parameter)
{
	M_UINT64 _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION_FLUX *_pointer = NULL;
	if ((_pointer = (FP_PRECISION_FLUX *)calloc(_dim1, sizeof(FP_PRECISION_FLUX))) == NULL) {
		if (strcmp(_parameter, CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_Flux\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_Flux\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer));
	_dataNumber = _dataNumber + _dim1;

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION_FLUX));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter, CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION_FLUX*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

FP_PRECISION_FLUX *FreeMem_FPFlux_I(FP_PRECISION_FLUX *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

double *AllocMem_FP64_I(int _dim1, char *_parameter)
{
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	double *_pointer = NULL;
	if ((_pointer = (double *)calloc(_dim1, sizeof(double))) == NULL) {
		if (strcmp(_parameter, CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP64_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP64_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer));
	_dataNumber = _dataNumber + _dim1;

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(double));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter, CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "double*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

FP_DOUBLE *AllocMem_FPDouble_I(int _dim1, char *_parameter)
{
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_DOUBLE *_pointer = NULL;
	if ((_pointer = (FP_DOUBLE *)calloc(_dim1, sizeof(FP_DOUBLE))) == NULL) {
		if (strcmp(_parameter, CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP64_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP64_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer));
	_dataNumber = _dataNumber + _dim1;

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_DOUBLE));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter, CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "double*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

FP_DOUBLE *FreeMem_FPDouble_I(FP_DOUBLE *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	if (_pointer != NULL) {
		free(_pointer);
	}
	
	_pointer = NULL;
	return _pointer;
}
/**
* @ Function: FreeMem_FP_I()
*   introduction: free memory for FP_PRECISION *pointer
* @ Parameter List: _pointer - memory pointer, need to be freed
*                   _parameter - parameter name
*   Return: FP_PRECISION *ponter
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
FP_PRECISION *FreeMem_FP_I(FP_PRECISION *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	if (_pointer != NULL) {
		free(_pointer);
	}
	
	_pointer = NULL;
	return _pointer;
}

double *FreeMem_FP64_I(double *_pointer, char *_parameter)
{
	if (strcmp(_parameter, CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

/**
* @ Function: AllocMem_FPGeo_I()
*   introduction: allocate memory for FP_PRECISION_GEO *pointer
* @ Parameter List: _dim1 - size of dimension 1
*                   _parameter - parameter name
*   Return: FP_PRECISION_GEO *ponter
* @ Created     : 2017/06/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
FP_PRECISION_GEO *AllocMem_FPGeo_I(int _dim1, char *_parameter)
{
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION_GEO *_pointer = NULL;
	if ((_pointer = (FP_PRECISION_GEO *)calloc(_dim1, sizeof(FP_PRECISION_GEO))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPGeo_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPGeo_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer));
	_dataNumber = _dataNumber + _dim1;

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION_GEO));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION_GEO*", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	}
	return _pointer;
}

/**
* @ Function: FreeMem_FPGeo_I()
*   introduction: free memory for FP_PRECISION_GEO *pointer
* @ Parameter List: _pointer - memory pointer, need to be freed
*                   _parameter - parameter name
*   Return: FP_PRECISION_GEO *ponter
* @ Created     : 2017/06/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
FP_PRECISION_GEO *FreeMem_FPGeo_I(FP_PRECISION_GEO *_pointer, char *_parameter)
{
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

FP_PRECISION **AllocMem_FP_II(int _dim1, int _dim2, char *_parameter)
{
	int _iter1;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION **_pointer = NULL;
	if ((_pointer = (FP_PRECISION **)calloc(_dim1, sizeof(FP_PRECISION *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_II\" - %s", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_II\" - local variable");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (FP_PRECISION *)calloc(_dim2, sizeof(FP_PRECISION))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_II\" - %s[%d]", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_II\" - local variable [%d]", _iter1);
			}
			exit(1);
		}
		_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber = _dataNumber + _dim2;
	}

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}

FP_DOUBLE **AllocMem_FPDouble_II(int _dim1, int _dim2, char *_parameter)
{
	int _iter1;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_DOUBLE **_pointer = NULL;
	if ((_pointer = (FP_DOUBLE **)calloc(_dim1, sizeof(FP_DOUBLE *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPDouble_II\" - %s", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPDouble_II\" - local variable");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (FP_DOUBLE *)calloc(_dim2, sizeof(FP_DOUBLE))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPDouble_II\" - %s[%d]", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPDouble_II\" - local variable [%d]", _iter1);
			}
			exit(1);
		}
		_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber = _dataNumber + _dim2;
	}

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_DOUBLE));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}

FP_PRECISION **FreeMem_FP_II(FP_PRECISION **_pointer, int _dim1, char *_parameter)
{
	int _iter1;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}

	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
        if (_pointer[_iter1] != NULL) {
            free(_pointer[_iter1]);
            _pointer[_iter1] = NULL;
        }
    }
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

FP_PRECISION **AllocMem_FP_IP(int _dim1, int *_dim2, char *_parameter)
{
	int _iter1;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION **_pointer = NULL;
	if ((_pointer = (FP_PRECISION **)calloc(_dim1, sizeof(FP_PRECISION *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_IP\" - %s", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_IP\" - local variable");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (FP_PRECISION *)calloc(_dim2[_iter1], sizeof(FP_PRECISION))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_IP\" - %s[%d]", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_IP\" - local variable [%d]", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber = _dataNumber + _dim2[_iter1];
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}
FP_PRECISION **FreeMem_FP_IP(FP_PRECISION **_pointer, int _dim1, char *_parameter)
{
	int _iter1;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}

	for (_iter1 = 0; _iter1<_dim1; _iter1++)
		free(_pointer[_iter1]);
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}

FP_PRECISION **AllocMem_FP_SpheHarmCoeff(int _numTrSet, int *_numAzimAng, int *_numPolAng, int _scattOrder, char *_parameter)
{
	int _iter1, _num;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION **_pointer = NULL;
	if ((_pointer = (FP_PRECISION **)calloc(_numTrSet, sizeof(FP_PRECISION *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_SpheHarmCoeff\" - %s", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_SpheHarmCoeff\" - local variable");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_numTrSet; _iter1++) {
		_num = _numAzimAng[_iter1] * _numPolAng[_iter1] * (_scattOrder + 1) * (_scattOrder + 1);
		if ((_pointer[_iter1] = (FP_PRECISION *)calloc(_num, sizeof(FP_PRECISION))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_SpheHarmCoeff\" - %s[%d]", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_SpheHarmCoeff\" - local variable [%d]", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber = _dataNumber + _num;
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}



FP_PRECISION_GEO **AllocMem_FPGeo_IP(int _dim1, int *_dim2, char *_parameter)
{
	int _iter1;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION_GEO **_pointer = NULL;
	if ((_pointer = (FP_PRECISION_GEO **)calloc(_dim1, sizeof(FP_PRECISION_GEO *))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPGeo_IP\" - %s", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPGeo_IP\" - local variable");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (FP_PRECISION_GEO *)calloc(_dim2[_iter1], sizeof(FP_PRECISION_GEO))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPGeo_IP\" - %s[%d]", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FPGeo_IP\" - local variable [%d]", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer[_iter1]));
		_dataNumber = _dataNumber + _dim2[_iter1];
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION_GEO));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION_GEO**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}
FP_PRECISION_GEO **FreeMem_FPGeo_IP(FP_PRECISION_GEO **_pointer, int _dim1, char *_parameter)
{
	int _iter1;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	}

	for (_iter1 = 0; _iter1<_dim1; _iter1++)
		free(_pointer[_iter1]);
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}


FP_PRECISION ***AllocMem_FP_III(int _dim1, int _dim2, int _dim3, char *_parameter)
{
	int _iter1, _iter2;
	long _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0.;
	FP_PRECISION _pointerSizeMB = 0.;
	FP_PRECISION ***_pointer = NULL;
	if ((_pointer = (FP_PRECISION ***)calloc(_dim1, sizeof(FP_PRECISION **))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_III\" - %s", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_III\" - local variable");
		}
		fflush(stdout);
		exit(69);
	}
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		if ((_pointer[_iter1] = (FP_PRECISION **)calloc(_dim2, sizeof(FP_PRECISION *))) == NULL) {
			if (strcmp(_parameter,CHAR_NONE) != 0) {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_III\" - %s[%d]", _parameter, _iter1);
			}
			else {
				printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_III\" - local variable [%d]", _iter1);
			}
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer[_iter1]));
		for (_iter2 = 0; _iter2<_dim2; _iter2++) {
			if ((_pointer[_iter1][_iter2] = (FP_PRECISION *)calloc(_dim3, sizeof(FP_PRECISION))) == NULL) {
				if (strcmp(_parameter,CHAR_NONE) != 0) {
					printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_III\" - %s[%d][%d]", _parameter, _iter1, _iter2);
				}
				else {
					printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMem_FP_III\" - local variable [%d][%d]", _iter1, _iter2);
				}
				fflush(stdout);
				exit(69);
			}
			_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_pointer[_iter1][_iter2]));
			_dataNumber = _dataNumber + _dim3;
		}
	}
	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(FP_PRECISION));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION***", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
        #endif
	return _pointer;
}
FP_PRECISION ***FreeMem_FP_III(FP_PRECISION ***_pointer, int _dim1, int _dim2, char *_parameter)
{
	int _iter1, _iter2;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		for (_iter2 = 0; _iter2<_dim2; _iter2++) {
			free(_pointer[_iter1][_iter2]);
            _pointer[_iter1][_iter2] = NULL;
		}
		free(_pointer[_iter1]);
        _pointer[_iter1] = NULL;
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}
int ***FreeMem_int_IPP(int ***_pointer, int _dim1, int *_dim2, char *_parameter)
{
	int _iter1, _iter2;
	if (strcmp(_parameter,CHAR_NONE) != 0)
        #ifndef UNITTEST
		OutputFreeMemory(_parameter);
        #endif
	for (_iter1 = 0; _iter1<_dim1; _iter1++) {
		for (_iter2 = 0; _iter2<_dim2[_iter1]; _iter2++) {
			free(_pointer[_iter1][_iter2]);
		}
		free(_pointer[_iter1]);
	}
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}
