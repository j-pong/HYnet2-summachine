# HYnet2-summachine
Summarization machine for multilingual paragraphs and sentences using hybrid seq2seq ASR model
## Installation

### 1. Installaion of docker for ssh env
- Basic of docker installation
https://colab.research.google.com/drive/1YhIBX9i59RN_9HEMihJX6TnFm9G5a7UL?authuser=1#scrollTo=swQ7g70S9O4J

- Docker for HYnet
```bash
cd docker

make _build

make run
```

### 2. Installaion of kaldi for utils
```bash
git clone https://github.com/kaldi-asr/kaldi kaldi

cd kaldi/tools
extras/install_mkl.sh -s
extras/check_dependencies.sh
make -j 28
extras/install_irstlm.sh

cd ../src/
./configure
make depend -j 28
make -j 28
```

### 3. Installaion of espnet for input pipelines
```bash
cd tools

./meta_installers/install_espnet.sh
```

### 4. Installaion of hynet for customizing egs
```bash
cd tools

./meta_installers/install_hynet.sh
```

## Run example
1. Set the DB path of egs/linersum/asr1/db.sh (i.e. LINERSUM)
2. Set the DB path and run it with the following lines:

```bash
cd egs/linersum/asr1

# training
./run.sh --asr_tag {tag_name}

# training with stage
./run.sh --asr_tag {tag_name} --stage 2

```

## Bugfix

matplotlib version error
- Remove matplotlib at tools/espnet/setup.py that is included in requirements