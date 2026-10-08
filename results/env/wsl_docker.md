# WSL2 и Docker

Дистрибутив `radar` (Ubuntu 24.04.5 LTS), диск `D:/VScode_projects/assets/wsl/radar/ext4.vhdx`, swap `D:/VScode_projects/assets/wsl/swap.vhdx`. `C:/Users/PC/.wslconfig`: memory 20GB, processors 10, swap 16GB. Docker Engine внутри дистрибутива, без Docker Desktop, плюс NVIDIA Container Toolkit.

Воспроизвести:

```
wsl.exe --install Ubuntu-24.04 --location D:\VScode_projects\assets\wsl\radar --name radar --no-launch
wsl.exe -d radar -u root --cd /mnt/d/VScode_projects/ru-it-jobs-radar -- bash scripts/wsl_provision.sh
wsl.exe --terminate radar
```

Вывод проверки (2026-10-08, `logs/wsl_docker_smoke.log`):

```
## whoami
radar
## docker version
client 29.8.2 server 29.8.2
Docker Compose version v5.6.0
## docker run hello-world
Unable to find image 'hello-world:latest' locally
Digest: sha256:5e23090353324d887c48ad5e5c56d294eab81588df9605b07d1afe895f9cc8f8
Status: Downloaded newer image for hello-world:latest
Hello from Docker!
This message shows that your installation appears to be working correctly.
To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.
To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash
Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/
For more examples and ideas, visit:
 https://docs.docker.com/get-started/
## docker run --gpus all nvidia/cuda nvidia-smi
Unable to find image 'nvidia/cuda:12.4.1-base-ubuntu22.04' locally
Digest: sha256:0f6bfcbf267e65123bcc2287e2153dedfc0f24772fb5ce84afe16ac4b2fada95
Status: Downloaded newer image for nvidia/cuda:12.4.1-base-ubuntu22.04
Thu Oct  8 10:20:00 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.54                 Driver Version: 595.79         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070        On  |   00000000:0A:00.0  On |                  N/A |
|  0%   47C    P8             28W /  240W |     935MiB /   8192MiB |     40%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              35      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+
```
