#ifndef FIXCACHE_H
#define FIXCACHE_H

class FixCache
{
public:
	__host__ __device__ FixCache();
	__host__ __device__ ~FixCache();
	__host__ __device__ void setFixResult(int row, int col, bool result);
	__host__ __device__ bool hasResult(int row, int col);
	__host__ __device__ bool fixResult(int row, int col);
	__host__ __device__ void init();

private:
	unsigned int m_definedTable[26];
	unsigned int m_fixTable[26];
};

#endif // FIXCACHE_H