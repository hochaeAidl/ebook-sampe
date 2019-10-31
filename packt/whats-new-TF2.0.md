# What's New in TensorFlow 2.0
### 


## 1 장. TensorFlow 2.0 시작하기  
이책의 목표는 독자들이 TensorFlow2.0에서 소개된 새로운 기능들에 익숙해지고 머신러닝 응용프로그램을 만들때 그 기능들의 잠재력을 활용할 수 있도록 돕는 것입니다.

이 장에서는 TF 2.0의 새로운 아키텍처 및 API 레벨의 변경 사항을 한눈에 볼 수 있습니다. TF 2.0 설치 및 설정을 다루고 Keras API 및 레이어 API와 같은 TensorFlow 1.x (TF 1.x)와 관련된 변경 사항을 비교합니다. 또한 TensorFlow Probability, Tensor2Tensor, Ragged Tensors와 같은 풍부한 확장 기능과 새롭게 추가된 loss function을 위한 custom training logic을 다룹니다.  

이 장에서는 또한, 레이어 API 및 기타 API의 변경 사항도 요약합니다.

이 장에서 다룰 내용은 다음과 같습니다.

    새로운 소식
    TF 2.0 설치 및 설정
    TF 2.0 사용
    풍부한 확장

### 기술적 요구 사항

다음 섹션에서 설명하는 단계를 시작하려면 다음이 필요합니다.

    파이썬 3.4 이상
    Ubuntu 16.04 이상이 설치된 컴퓨터 (macOS 또는 기타 Linux 변형과 같은 대부분의 * NIX 기반 시스템에서 사용되는 명령어들을 활용)

## What's new?
TF 2.0의 철학은 단순성과 사용 편의성을 기반으로합니다. 주요 업데이트에는 tf.keras를 사용하여 손쉬운 모델 빌드와 간결한 즉시 실행, 모든 플랫폼에 대한 프로덕션 및 상업적 사용을위한 강력한 모델 배포, 강력한 실험 기술 및 연구 도구,그리고 보다 직관적 인 API 구성을위한 API 단순화가 포함됩니다.

TF 2.0의 새로운 구성은 다음 다이어그램으로 단순화되었습니다.
![diagram](./z_image/diagram)

위의 다이어그램은 Python API를 사용한 교육 및 배포에 중점을 두고 있습니다. 그러나 동일한 프로세스에 Julia, JavaScript 및 R을 포함한 다른 지원 언어가 적용됩니다. TF 2.0의 흐름은 두 가지 섹션 (모델 훈련 및 모델 배포)으로 구분됩니다. 여기서 모델 훈련에는 데이터 파이프 라인, 모델 생성, 훈련, 그리고 유통 전략을 포함되고; 모델 배포에는 TF Serving, TFLite, TF.js 및 기타 언어 바인딩과 같은 다양한 배포 방법이 포함됩니다. 이 다이어그램의 구성 요소는 각각 해당 장에서 자세히 설명됩니다


TF 2.0의 가장 큰 변화는 즉시 실행(eager excution)의 추가입니다. 즉시 실행은 반드시 그래프를 작성하지 않고도 작업을 즉시 평가하는 명령형 프로그래밍 환경입니다. 모든 작업은 사용자가 나중에 계산할 수있는 계산 그래프를 구성하는 대신 구체적인 값을 반환합니다.

이를 통해 TensorFlow 모델을 구축하고 교육하는 것이 훨씬 쉬워지고 TF 1.x 코드로 인한 상용구 코드의 대부분이 줄어 듭니다. 즉시실행에는 표준 Python 코드 흐름을 따르는 직관적 인 인터페이스가 있습니다. pdb와 같은 디버깅 용 표준 Python 모듈을 사용하여 오류의 소스레벨에서 코드를 검사 할 수 있으므로 즉시실행으로 작성된 코드는 디버깅하기가 훨씬 쉽습니다. 자연스러운 Python 제어 흐름과 반복 지원으로 인해 사용자 지정 모델을 만드는 것도 더 쉽습니다.

TF 2.0의 또 다른 주요 변경 사항은 텐서 플로우 모델 생성 및 훈련을위한 표준 모듈로 tf.keras로 마이그레이션하는 것입니다. Keras API는 TF 2.0의 중앙 고급 중심 상위 API이므로 TensorFlow를 쉽게 시작할 수 있습니다. Keras는 딥 러닝 개념을 독립적으로 구현하지만 tf.keras 구현에는 즉각적인 반복 및 디버깅을위한 즉시실행과 같은 향상된 기능이 포함되어 있으며 확장 가능한 입력 파이프 라인 구축을 위한 tf.data도 포함되어 있습니다.

tf.keras의 예제 워크 플로우는 먼저 tf.data 모듈을 사용하여 데이터를로드하는 것입니다. 이를 통해 모든 데이터를 메모리에 저장하지 않고 디스크에서 대량의 데이터를 스트리밍 할 수 있습니다. 그런 다음 개발자는 tf.keras 또는 사전 작성된 측정기를 사용하여 모델을 빌드, 훈련 및 유효성 검증을 진행합니다. 다음 단계는 열악한 즉시실행의 이점을 사용하여 모델을 실행하고 디버깅하는 것입니다. 모델이 본격적인 훈련을 할 준비가되면 분산 훈련에 대한 분산 전략을 사용하십시오. 마지막으로 모델을 배포 할 준비가되면 다이어그램에 표시된 배포 전략을 통해 모델을 저장된 모델 SavedModel 모듈로 내보내 배포 할 수 있습니다.

## TF 1.x에서 변경된 사항
TF 1.x와 TF 2.0의 첫 번째 주요 차이점은 API 구성입니다. TF 2.0은 API 구조의 중복성을 줄였습니다. 주요 변경 사항에는 absl-py 및 내장 로깅 기능과 같은 다른 Python 모듈을 위해 tf.app, tf.flags 및 tf.logging 제거가 포함됩니다.

tf.contrib 라이브러리도 이제 기본 TensorFlow 저장소에서 제거되었습니다. 이 라이브러리에서 구현 된 코드가 다른 위치로 이동되었거나 TensorFlow 애드온 라이브러리로 이동되었습니다. 이러한 이동의 이유는 contrib 모듈이 단일 리포지토리에서 유지 관리 할 수있는 수준 이상으로 성장했기 때문입니다

다른 변경 사항으로는 tf.data 사용을 선호하는 사용을 위하여 QueueRunner 모듈 제거, 그래프 collection 제거, 그리고 변수 처리 방법 변경 등이 있습니다. QueueRunner 모듈은 훈련을 위해 모델에 데이터를 제공하는 방법 이었지만 tf.data보다 사용이 훨씬 복잡하고 어렵습니다. 따라서 이제는 tf.data 가 data 제공의 기본설정이 되었습니다. 데이터 파이프 라인에 tf.data를 사용하면 얻을 수있는 다른 이점은 3 장, 입력 데이터 파이프 라인 디자인 및 구성에 설명되어 있습니다.

TF 2.0의 또 다른 주요 변경 사항은 더 이상 전역 변수가 없다는 것입니다. TF 1.x에서 tf.Variable을 사용하여 생성 된 변수는 기본 그래프에 표시되고 이름을 통해 여전히 복구 가능합니다. TF 1.x는 사용자가 변수 범위, 전역 컬렉션, 그리고  tf.get_global_step 및 tf.global_variables_initializer와 같은 도우미 메서드와 같은 변수를 복구 할 수 있도록 모든 종류의 메커니즘을 가졌습니다. 제공합니다. 이 모든 것은 파이썬에서 기본 변수 동작을 위해 TF 2.0에서 제거되었습니다.

---
## TF 2.0 설치 및 설정
이 섹션에서는 TF2.0을 여러분의 시스템에 설치하는 몇가지 방법과 시스템 구성에 대하여 설명합니다.
엔트리 레벨 사용자는 pip 및 virtualenv 기반 방법으로 시작하는 것이 좋습니다. GPU 버전 사용자에게는 ‘도커' 사용을 권장합니다.

### pip 설치 및 사용
초심자에게, pip는 Python 커뮤니티에서 널리 사용되는 패키지 관리 시스템입니다. 시스템에 설치되어 있지 않으면 계속 진행하기 전에 설치하십시오. 많은 Linux 설치에서 배포판에는 Python 및 pip가 기본적으로 설치됩니다. 다음 명령을 입력하여 pip가 설치되어 있는지 확인할 수 있습니다.
```
python3 -m pip --help
```

이때 pip이 지원하는 다른 명령어들에 대한 설명이 짧게 나온다면 pip이 설치되어 있는 것입니다. 설치되어 있지 않다면, pip이라는 이름을 가진 module이 없다는 오류 메시지가 출력됩니다.

**Tip**  
일반적으로 개발 환경을 격리하는 것이 좋습니다. 이는 종속성 관리를 크게 단순화하고 소프트웨어 개발 프로세스를 간소화 합니다. Python에서 virtualenv라는 도구를 사용하여 환경을 격리 할 수 있습니다. 이 단계는 선택 사항이지만 강력히 권장됩니다.


```
>> mkdir .venv
>> virtualenv --python=python3.6 .venv/
>> source .venv/bin/activate
```

다음 명령과 같이 pip를 사용하여 TensorFlow를 설치할 수 있습니다.
```
pip3 install tensorflow==version_tag
```
예를 들어, 버전 2.0.0-beta1을 설치하려는 경우 명령은 다음과 같아야합니다. 같습니다.
```
pip3 install tensorflow == 2.0.0-beta1
```

최신 패키지 업데이트의 전체 목록은 다음 링크에서 확인하실 수 있습니다.   
[https://pypi.org/project/tensorflow/#history](https://pypi.org/project/tensorflow/#history)

다음 명령을 실행하여 설치를 테스트 할 수 있습니다.
```
python3 -c "import tensorflow as tf; a = tf.cotant(1); print(tf.math.add(a, a))"
```

### Using Docker
TensorFlow의 설치를 시스템의 다른부분으로 부터 분리하려면, Docker image를 이용하는 것을 고려해야만 합니다. 이를 위해서는 시스템에 Docker가 우선 설치되어 있어야 합니다. Docker 설치에 대한 정보는 아래 링크를 참고하십시오.

[https://docs.docker.com/install/](https://docs.docker.com/install/)

**Tip**   
Linux system에서 **sudo** 없이 Docker 를 사용하려면 아래 링크의 'post-install' 과정을 진행하십시오.

[https://docs.docker.com/install/linux/linux-postinstall/](https://docs.docker.com/install/linux/linux-postinstall/)

The TensorFlow team officially supports Docker images as a mode of installation. To the user, one implication of this is that updated Docker images will be made available for download at https://hub.docker.com/r/tensorflow/tensorflow/.

TensorFlow team은 공식적으로 Docker image를 설치 모드로 제공합니다. 사용자에게 이것은 update된 image는 아래 링크에서 download 가능하다는 의미입니다.

[https://hub.docker.com/r/tensorflow/tensorflow/](https://hub.docker.com/r/tensorflow/tensorflow/)

아래 명령으로 Docker image를 내려받으십시오.
```
docker pull tensorflow/tensorflow:YOUR_TAG_HERE
```

The previous command should've downloaded the Docker image from the centralized repository. To run the code using this image, you need to start a new container and type the following:  
위 명령은 Docker image를  내려받을 것입니다. 내려받은 이미지를 가지고 코드를 수행하기 위해서는 새로운 container 기 필요합니다. 아래 명령을 수행하십시오.
```
docker run -it --rm tensorflow/tensorflow:YOUR_TAG_HERE 
python -c "import tensorflow as tf; a = tf.constant(1); print(tf.math.add(a, a))"
```

A Docker-based installation is also a good option if you intend to use GPUs. Detailed instructions for this are provided in the next section. 
GPU를 사용하고자 하는 경우, Docker 를 기반으로 하는 것은 좋은 방법입니다.

## GPU installation
Installing the GPU version of TensorFlow is slightly different from the process for the CPU version. It can be installed using both pip and Docker. The choice of installation process boils down to the end objective. The Docker-based process is easier as it involves installing fewer additional components. It also helps avoid library conflict. This can, though, introduce an additional overhead of managing the container environment. The pip-based version involves installing more additional components but offers a greater degree of flexibility and efficiency. It enables the resultant installation to run directly on the local host without any virtualization.

TensorFlow의 GPU 버젼을 설치하는 것은 CPU 버젼을 설치하는 것과는 약간 다릅니다. pip 과 Docker로 설치할 수 있습니다. 설치과정에 대한 선택은 결국 최종 결과로 연결됩니다. Docker 기반의 과정은 최소의 추가 요소 설치를 동반하고 있어서 보다 쉽습니다. 또한 라이브러리 충돌을 피할 수도 있습니다. 그렇지만 이 방법은 컨테이너 환경관리를 위한 추가 오버헤드를 수반할 수 있습니다. pip 기반의 설치는 좀더 많은 추가 요소들의 설치를 동반하지만 보다 나은 유연성과 효율성을 제공합니다. 이 방법은 설치를 virtualization 없이 직접 local host에서 수행할 수 있도록 합니다.


To proceed, assuming you have the necessary hardware set up, you would need the following piece of software at a minimum. Detailed instructions for installation are provided in the link for NVIDIA GPU drivers (https://www.nvidia.com/Download/index.aspx?lang=en-us).

진행하려면, 필요한 하드웨어 셋업이 되어 있다고 가정하고, 다음 소프트웨어를 잠깐 수행할 필요가 있습니다. 자세한 설명은 아래 NVIDI GPU driver 링크를 참고하십시오.
[https://www.nvidia.com/Download/index.aspx?lang=en-us](https://www.nvidia.com/Download/index.aspx?lang=en-us)


## Docker로 설치
At the time of writing this book, this option is only available for NVIDIA GPUs running on Linux hosts. If you meet the platform constraints, then this is an excellent option as it significantly simplifies the process. It also minimizes the number of additional software components that you need to install by leveraging a pre-built container. To proceed, we need to install nvidia-docker. Please refer the following links for additional details:   
이 책을 쓸 당시에는 이 옵션이 Linux host에서 돌고있는 NVIDIA GPU들에 대하여서만 가능하였다. 플랫폼 제약에 직면하면, 이 옵션은 프로세스를 매우 간다한게 하기때문에 매우 유용하다. 또 이 옵션은 미리 만들어진 컨테이너를 이용하여 설치한 추가 소프트웨어 요소들을 최소화 합니다. 다음단계로 넘어가기 위해서는 nvidia-docker를 설치하여야 합니다. 더 자세한 사항은 아래 링크를 참고하십시오.

Installation: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)   
FAQs: [https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#platform-support](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#platform-support)

Once you've completed the steps described in the preceding links, take the following steps:   
위의 링크들에서 설명된 절차를 마치고 나면, 아래 절차를 진행하십시오.

1. GPU 사용이 가능한지 확인합니다: Test whether the GPU is available:
```
lspci | grep -i nvidia
```
2. nvidia-docker 설치를 확인합니다.(nvidia-docker v2일 경우) Verify your nvidia-docker installation (for v2 of nvidia-docker):
```
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```
3. Docker image 를 내려 받습니다: Download a Docker image locally:
```
docker pull tensorflow/tensorflow:YOUR_TAG_HERE
```
4. Let's say you're trying to run the most recent version of the GPU-based image. You'd type the following:   
가장 최신 버젼을 사용하려면 아래 와 같이 하면 됩니다.
```
docker pull tensorflow/tensorflow:latest-gpu
```
5. 컨테이너를 시작하고 코드를 실행합니다. Start the container and run the code:
```
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu    python -c "import tensorflow as tf; a = tf.constant(1); print(tf.math.add(a, a))"
```

## pip로 설치 / Installing using pip
If you would like to use TensorFlow with an NVIDIA GPU, you need to install the following additional pieces of software on your system. Detailed instructions for installation are provided in the links shared:   
NVIDIA GPU와 함께 TensorFlow 를 사용하고 싶다면, 다음 추가 사항을 설치하여야 합니다.

    CUDA Toolkit: TensorFlow supports CUDA 10.0 (https://developer.nvidia.com/cuda-toolkit-archive)
    CUPTI ships with the CUDA Toolkit (https://docs.nvidia.com/cuda/cupti/)
    The cuDNN SDK (version 7.4.1 or above) (https://developer.nvidia.com/cudnn)
    (Optional) TensorRT 5.0 to improve latency and throughput for inference on some models (https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)

Once all the previous components have been installed, this is a fairly straightforward process.    
위 요소들을 다 설치하고 나면, 이제는 매우 간단한 과정입니다.

pip로 TensorFlow 설치: / Install TensorFlow using pip:
```
pip3 install tensorflow-gpu==version_tag
```
For example, if you want to install tensorflow-2.0:alpha, then you'd have to type in the following command:   
예를들어, tensorflow-2.0:alpha를 설치하려면, 아래 명령어를 수행한다.
```
pip3 install tensorflow-gpu==2.0.0-alpha0
```
A complete list of the most recent package updates is available at https://pypi.org/project/tensorflow/#history.

가장최근 package update들의 모든 리스트는 아래 link에서 찾을 수 있습니다.   
[https://pypi.org/project/tensorflow/#history](https://pypi.org/project/tensorflow/#history)

You can test your installation by running the following command:
아래 명령으로 설치를 확인할 수 있습니다.
```
python3 -c "import tensorflow as tf; a = tf.constant(1); print(tf.math.add(a, a))"
```
