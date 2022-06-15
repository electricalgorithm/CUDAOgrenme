#include <stdio.h>

__global__ void vector_addition(int *A,int *B,int *C,int size)//CUDA kernel
{
        // Thread ID'mizi alıyoruz.
        int id = (blockIdx.x * blockDim.x) + threadIdx.x;

        // Eğer Thread ID'si size'ı geçmiyorsa toplamayı yapıyoruz. 
        if (id < size)
                C[id] = A[id] + B[id];
}

int main()
{
        int size = 10000002;            //Dizi büyüklüğü (2.günki soru için 3'e tam bölünmeli)
        int ThreadPerBlock = 1024;      //Blok büyüklüğü. 32'nin katı olması iyi olur
        int BlockSize = (int) ceil((float)size/ThreadPerBlock); //@ Blok sayısı hesaplanacak
        printf("BlockSize = %d\n", BlockSize);

        int *A_Host,*B_Host,*C_Host;
        A_Host = new int[size];         //CPU belleğinde (Heap bölgesi) yer açılıyor
        B_Host = new int[size];         //CPU belleğinde (Heap bölgesi) yer açılıyor
        C_Host = new int[size];         //CPU belleğinde (Heap bölgesi) yer açılıyor

        for (int i = 1; i <= size; i++) //Diziye başlangıç değerleri atanıyor
        {
                A_Host[i-1] = i;
                B_Host[i-1] = 0;
        }

        int *A_GPU, *B_GPU, *C_GPU;
        //@ GPU Ana belleginde yer ayırılacak
        cudaMalloc(&A_GPU, sizeof(int)*size);
        cudaMalloc(&B_GPU, sizeof(int)*size);
        cudaMalloc(&C_GPU, sizeof(int)*size);

        //@ Blok Büyüklüğü ve Grid Büyüklüğü dim3 türünde tanımlanacak
        dim3 DimBlock(ThreadPerBlock);
        dim3 DimGrid(BlockSize);

        cudaEvent_t start, stop;        //Süre değişkenleri
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float totaltime;                //Toplam süre değişkeni

        cudaEventRecord(start); //Süre başlatıldı
	
	//@ CPU belleğinden GPU ana bellegine veri transferi gerçekleştirilecek
        cudaMemcpy(A_GPU, A_Host, sizeof(int)*size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_GPU, B_Host, sizeof(int)*size, cudaMemcpyHostToDevice);

        //@ CUDA Kernel çalıştırılacak
        vector_addition<<<DimGrid, DimBlock>>>(A_GPU, B_GPU, C_GPU, size);

        //@ GPU ana belleğinden CPU bellegine veri transferi gerçekleştirilecek
        cudaMemcpy(C_Host, C_GPU, sizeof(int)*size, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);          //Süre durduruldu
        cudaEventSynchronize(stop);     //Event işlemleri bitene kadar program beklemekte
        cudaEventElapsedTime(&totaltime, start, stop);  //Geçen süre hesaplanıyor
        printf("%f\n", totaltime);
        printf("%d\n", C_Host[size-1]);

        delete[] A_Host;        //Dizi CPU belleğinden siliniyor
        delete[] B_Host;        //Dizi CPU belleğinden siliniyor
        delete[] C_Host;        //Dizi CPU belleğinden siliniyor

        //@ Diziler GPU ana belleğinden silinecek
        cudaFree(A_GPU);
        cudaFree(B_GPU);
        cudaFree(C_GPU);

        cudaError_t err = cudaGetLastError();//GPU'da oluşan son hatayı yakalıyor
        if ( err != cudaSuccess )
                printf("CUDA Error: %s\n",cudaGetErrorString(err));
}
