# self-long-instruct
The project to enhance the self-instruct method to long-context instruction tuning dataset auto-generation based on long-LLMs, Retrieval-Augmented Generation (RAG) and LLMs-as-Agents


### Base

* **self-instruct**: 
  * paper link: [here](https://arxiv.org/abs/2212.10560) | github link: [here](https://github.com/yizhongw/self-instruct)
* **Retrieval-Augmented Generation (RAG)**: 
  * paper link: [here](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) | github link: [here](https://github.com/langchain-ai/langchain)
* **LLMs-as-Agents**:
  * paper link: [here](https://arxiv.org/abs/2308.08155) | github link: [here](https://github.com/microsoft/autogen)



### Preparation

* install the pip dependences:
    ```sh
    pip install -r requirements.txt
    ```
* download the punkt from nltk :
  * method1: download through the api
    ```sh
    import nltk
    nltk.download('punkt')
    ```
  * method2: if the api fails, you can go to the [github repo](https://github.com/nltk/nltk_data) and follow the steps below:
    * step1: download the whole `packages` directory into your conda env path like `/home/user/anaconda3/envs/myenv/` and rename it `nltk_data`
    * step2: unzip the zip files through the `nltk_data`, especially the `tokenizers/` and `taggers/`, and to make it convenient, we also provide a function to do it automatically:
      ```sh
      from src.utils import unzip_nltk_data
      nltk_data_dir = "/home/user/anaconda3/envs/myenv/"
      unzip_nltk_data(nltk_data_dir, remove=True) 
      ```
* install the poppler tools to make `pdf2image` work (*Assuming your OS is Linux, well if not, you can check [pdf2image installation guide](https://pdf2image.readthedocs.io/en/latest/installation.html) further*):
    ```sh
    sudo apt-get install poppler-utils
    ```
* follow the guide [here](https://www.libreoffice.org/get-help/install-howto/) and install the `LibreOffice` tool to make `unstructured.partition.doc` work