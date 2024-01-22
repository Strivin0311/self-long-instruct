import sys
sys.path.append("..")

import os
import time
from tqdm import tqdm
from typing import List, Dict, Union, Optional

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, UnstructuredFileLoader, TextLoader
from langchain.document_loaders.word_document import Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import suppress_stderr, info_str


def load_doc(
            file_path: Union[Optional[str], List[str]] = None,
            dir_path: Optional[str] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 0,
            unstructured: bool = True,
            verbose: bool = True,
            ) -> List[Document]:
    """ load the docx files from the (list of) file path or all ones in the directory 
        and return a list of documents
    Args:
        file_path (str): the (list of) file path(s), if this argument is None, then load all docx files in the directory pointed by dir_path.
        dir_path (str): the directory path, if this argument is None, then load all docx files pointed by file_path(s).
        chunk_size (int): the size of each document chunk.
        chunk_overlap (int): the overlap between every adjacent document chunk.
        unstructured (bool): if True, use UnstructuredWordDocumentLoader to load the docx files clearer but slightly slower, otherwise use Docx2txtLoader
        verbose (bool, optional): whether to print the information. Defaults to True.

    Returns:
        List: a list of documents
    """
    
    # create the doc loaders
    if file_path is not None:
        if not isinstance(file_path, list): file_path = [file_path]
        doc_loaders = [
            Docx2txtLoader(file_path) if not unstructured else \
            UnstructuredWordDocumentLoader(file_path, mode='elements', strategy='fast')
            for file_path in file_path
        ]
    elif dir_path is not None:
        doc_loaders = [
            Docx2txtLoader(os.path.join(dir_path, filename)) if not unstructured else \
            UnstructuredWordDocumentLoader(os.path.join(dir_path, filename), mode='elements', strategy='fast')
            for filename in os.listdir(dir_path) if filename.endswith(".doc") or filename.endswith(".docx")
        ]
    else: raise ValueError("Either file_path or dir_path must be provided")
    
    # define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # load and split the doc files into documents
    docs = []
    if verbose: print(info_str(f"Loading from {len(doc_loaders)} doc files"))
    for loader in tqdm(doc_loaders, total=len(doc_loaders)): 
        with suppress_stderr(): # FIXME: avoid the "Multiple definitions in dictionary" error when loading the Chinese doc files
            # docs.extend(loader.load())
            docs.extend(loader.load_and_split(text_splitter))
    if verbose: print(info_str(f"Loaded {len(docs)} documents"))
    return docs


def load_pdf(
            file_path: Union[Optional[str], List[str]] = None,
            dir_path: Optional[str] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 0,
            unstructured: bool = True,
            verbose: bool = True,
            ) -> List[Document]:
    """ load the pdf files from the (list of) file path or all ones in the directory 
        and return a list of documents
    Args:
        file_path (str): the (list of) file path(s), if this argument is None, then load all pdf files in the directory pointed by dir_path.
        dir_path (str): the directory path, if this argument is None, then load all pdf files pointed by file_path(s).
        chunk_size (int): the size of each document chunk.
        chunk_overlap (int): the overlap between every adjacent document chunk.
        unstructured (bool): if True, use UnstructuredPDFLoader to load the docx files clearer but slightly slower, otherwise use PyPDFLoader.
        verbose (bool, optional): whether to print the information. Defaults to True.

    Returns:
        List: a list of documents
    """
    import warnings
    warnings.filterwarnings("ignore", module="pypdf")
    
    # create the pdf loaders
    if file_path is not None:
        if not isinstance(file_path, list): file_path = [file_path]
        pdf_loaders = [
            PyPDFLoader(file_path) if not unstructured else \
            UnstructuredPDFLoader(file_path, mode='elements', strategy='fast')
            for file_path in file_path
        ]
    elif dir_path is not None:
        pdf_loaders = [
            PyPDFLoader(os.path.join(dir_path, filename)) if not unstructured else \
            UnstructuredPDFLoader(os.path.join(dir_path, filename), mode='elements', strategy='fast')
            for filename in os.listdir(dir_path) if filename.endswith(".pdf")
        ]
    else: raise ValueError("Either file_path or dir_path must be provided")
    
    # define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # load and split the pdf files into documents
    docs = []
    if verbose: print(info_str(f"Loading from {len(pdf_loaders)} pdf files"))
    for loader in tqdm(pdf_loaders, total=len(pdf_loaders)): 
        with suppress_stderr(): # FIXME: avoid the "Multiple definitions in dictionary" error when loading the Chinese pdf files
            # docs.extend(loader.load())
            docs.extend(loader.load_and_split(text_splitter))
    if verbose: print(info_str(f"Loaded {len(docs)} documents"))
    return docs


def load_string(
        string: str,
        unstructured: bool = True,
        return_str: bool = False,
    ) -> Union[str, List[Document]]:
    tmp_file_path = f"./tmp_{time.time()}.txt"
    with open(tmp_file_path, 'w', encoding='utf-8') as f:
        f.write(string)
    
    loader = UnstructuredFileLoader(tmp_file_path) if unstructured else TextLoader(tmp_file_path)
    
    docs = loader.load()
    
    os.remove(tmp_file_path)
    
    if return_str:
        new_string = "\n".join([doc.page_content for doc in docs])
        return new_string
    
    return docs
    