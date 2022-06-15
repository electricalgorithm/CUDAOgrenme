#!/bin/bash
#SBATCH --account=egitim #Egitimden sonra kendi TRUBA hesabınızla çalıştırmak için bu satırı silin
#SBATCH --reservation=akya  #Egitimden sonra kendi TRUBA hesabınızla çalıştırmak için bu satırı silin
#SBATCH -p akya-cuda        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A egitim       # Kullanici adi
#SBATCH -J Lab1        # Gonderilen isin ismi
#SBATCH -o Lab1.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 10  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=00:01:00      # Sure siniri koyun.

module load /truba/sw/centos7.3/modulefiles/centos7.3/lib/cuda/7.5

nvcc Lab1_Soru.cu -o Lab1_Soru
./Lab1_Soru

