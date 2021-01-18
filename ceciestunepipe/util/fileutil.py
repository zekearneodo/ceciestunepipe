import os
import sys
import glob
import logging
import shutil


logger = logging.getLogger('ceciestunepipe.util.fileutil')

# copy path to dest, skip f existed
def safe_copy(src_path: str, dest_path: str):
    if not os.path.exists(dest_path):
        shutil.copyfile(src_path, dest_path)
    else:
        logger.info('file {} already existed, nuttin to do'.format(dest_path))

# append one binary file to another, chunked
def append_binary(src_path: str, dest_path: str, chunk_size: int=4194304):
    logger.info('Appending binaries {} -> {}'.format(src_path, dest_path))
    with open(dest_path, "ab") as dest_file, open(src_path, "rb") as src_file:
        shutil.copyfileobj(src_file, dest_file, chunk_size)