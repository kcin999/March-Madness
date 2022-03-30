"""
This module deals with all things realted to the directories. Right now it contains folders and gets the current Directory.
This file can also be imported as a module and contains the following functions:
	* getCurrentDir - Gets the current Directory
	* createFolder - Creates the Folder
	* doesFileExist - checks to see if a file exists
	* deleteFile - deletes file if it does exist
"""

import os
import logging
import datetime

logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(message)s')

def getCurrentDir() -> str:
	"""Gets the current directory. Often called the root through this program
	
	Returns
	-------
	str
		The current Directory
	"""
	path = os.getcwd()
	return str(path)

def createFolder(path: str):
	"""Creates the folder if they have not been created.
	
	Parameters
	-----------
	path :  str
		The Path to the folder you would like to have created
	"""
	if os.path.isdir(path) == False:
		try:
			os.mkdir(path)
		except OSError as e:
			print(e)
			print("Creation of the directory %s failed" % path)
			logging.info("Creation of the directory %s failed" % path)
		else:
			print("Successfully created the directory %s" % path)

def doesFileExist(fileName: str) -> bool:
	"""Checks to see if a file exists
	
	Parameters
	----------
	fileName : str
		File name to check and see if exists
	
	Returns
	-------
	bool
		Returns True if exists, false if it does not
	"""
	return os.path.exists(fileName)

def deleteFile(fileName: str):
	if doesFileExist(fileName):
		os.remove(fileName)
	else:
		print(fileName + " does not exist")