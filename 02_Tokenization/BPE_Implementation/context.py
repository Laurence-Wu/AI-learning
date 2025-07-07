#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import unicode_literals, division
import os
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

@contextmanager
def read_files_from_folder(folder_path, encoding='utf-8'):
    """Read all files from a folder and yield their contents."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    yield f
            except (UnicodeDecodeError, PermissionError):
                continue