import sys
sys.path.append("..")

import os
from tqdm import tqdm
from typing import List, Dict, Union, Optional

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import suppress_stderr, info_str


def load_pdf(
            file_path: Union[Optional[str], List[str]] = None,
            dir_path: Optional[str] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 0,
            verbose: bool = True,
            ) -> List[Document]:
    """ load the pdf files from the (list of) file path or all ones in the directory 
        and return a list of documents
    Args:
        file_path (str): the (list of) file path(s), if this argument is None, then load all pdf files in the directory pointed by dir_path.
        dir_path (str): the directory path, if this argument is None, then load all pdf files pointed by file_path(s).
        chunk_size (int): the size of each document chunk.
        chunk_overlap (int): the overlap between every adjacent document chunk.
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
            PyPDFLoader(file_path) 
            for file_path in file_path
        ]
    elif dir_path is not None:
        pdf_loaders = [
            PyPDFLoader(os.path.join(dir_path, filename)) 
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
        