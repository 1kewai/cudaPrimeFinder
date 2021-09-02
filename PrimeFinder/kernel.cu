//CUDA Cのinclude
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//cpu側include
#include <stdio.h>
#include <stdlib.h>

//CUDA How-to memo
//スレッドの固有の番号を取得:blockIdx.x * blockDim.x + threadIdx.x
//メモリの内容をコピー:cudaMemcpy(HostMemory, Device_array, num_of_rnd * 4, cudaMemcpyDeviceToHost);
//CUDAの処理完了を待機:cudaDeviceSynchronize();
//デバイスメモリ確保:int* Device_addr; cudaMalloc((void**)&Device_addr, サイズ);
//デバイスメモリ解放;cudaFree(ポインタ)
//メモリコピー:cudaMemcpy(HostMemory(dst), Device_array(src), num_of_rnd * 4, cudaMemcpyDeviceToHost);

//素数を保存するためのテーブルのサイズ
const int table_size = 1024;
//素数テーブルの数
const int table_n = 512;
//結果をGPUとやり取りするためのフラグ変数のアドレス
short* flag;
short t = short(1);
short f = short(0);

//GPUコード
//渡された数値が指定されたアドレスの素数テーブルにある数値で割り切れるか確かめる。もし割り切れるなら結果フラグを1, 割り切れないなら0にする。
__global__ void Device_PrimeChk(long number, long* table, short* flag ){
	if (number % table[blockIdx.x * blockDim.x + threadIdx.x] == 0) {
		*flag = short(1);
	}
}

//渡されたアドレスから、渡された数だけlongの数を読み取ってprintf
__global__ void debug(long* table, int number) {
	for (int i = 0; i < number; i++) {
		printf("%ld\n", table[i]);
	}
}

//CPUコード
//テーブルを初期化する
//最初の中身は素数である2で初期化を行う
void Host_init_table(long* table) {
	for (int i = 0; i < table_size; i++) {
		table[i] = long(2);
	}
}

int main(){
	//GPUの素数テーブルのアドレステーブル部分をcpu側スタックに確保
	long* Device_table[table_n];

	//素数テーブルの雛形をcpuで作成、中身を2で埋める
	long* table_template;
	table_template = new long[table_size];
	Host_init_table(table_template);

	//GPU側にテーブルのテンプレートを必要数だけコピー
	for (int i = 0; i < table_n; i++) {
		cudaMalloc((void**)&Device_table[i], 8 * table_size);
		cudaMemcpy(Device_table[i], table_template, 8 * table_size, cudaMemcpyHostToDevice);
	}

	//GPU側に結果フラグを書き込んでいくメモリ領域を作成し０を書き込む
	cudaMalloc((void**)&flag, 2);
	cudaMemcpy(flag, (void**)&f, 2, cudaMemcpyHostToDevice);
	//CPU側からはテーブルのテンプレートを削除
	free(table_template);

	//実際に計算を行う。
	//GPU側でそれまで発見された素数を使って割り算=>もし成功したらcpuがそれをポーリング=>GPUメモリに書き込むことで素数リストを更新
	int page = 0;//素数を次に記録するページ
	int index = 1;//素数を次に記録する位置
	long search = 3;//次に素数かどうかを検証する位置
	
	int i = 0;
	bool chk = false;
	while (1) {
		chk = false;
		//ページ数が複数ある場合、すべてのページに対して割り切れる数が無いか探索を行う
		short Host_flag = f;
		for (i = 0; i <= page; i++) {
			//GPUに演算を投げる
			Device_PrimeChk << <1, table_size >> > (search, Device_table[i], flag);
			cudaDeviceSynchronize();
			cudaMemcpy(&Host_flag, flag, 2, cudaMemcpyDeviceToHost);
			if (Host_flag != f) {
				search++;
				chk = true;
				cudaMemcpy(flag, (void**)&f, 2, cudaMemcpyHostToDevice);
				continue;
			}
		}
		if (chk == false) {
			if (index == table_size) {
				index = 0;
				page++;
			}
			printf("%ld\n", search);
			cudaMemcpy(&Device_table[page][index], &search, 8, cudaMemcpyHostToDevice);
			index++;
			search++;
		}
	}
}