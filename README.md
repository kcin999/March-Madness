# March-Madness
This is a project with March Madness and NCAA data. Had the idea to go find data in class one day, and started creating various metrics.

The end goal is to be able to use data to help predict and make choosing brackets easier down the road. I will start encorporating past data soon. This is a hobby project of mine, so it will be done over time, but classes come first

## Installation
There are a couple of softwares / programs that needs to be installed. 
* Python
* Microsoft PowerBI

### Python
Python needs to be installed. Can be found here: https://www.python.org/downloads/

**Libraries Needed:**
* [sportspi](https://sportsreference.readthedocs.io/en/stable/)

Can be installed with pip:

```
pip install sportsipy
```

### Microsoft PowerBI
I am using Microsoft PowerBI to analyze my data. My sample file is in the directory called "". All files that end in "*.pbix" is a PowerBI file.

**Installing PowerBI**
An installer for Microsfot PowerBI can be found here: https://powerbi.microsoft.com/en-us/downloads/

**Other Notes**
For some reason or another, PowerBI does ***NOT*** support relative file paths. So unfortunatley, you will have to go into the PowerQuery Editor, and change the file path. 

#### Steps to change File Path
1. Open the .pbix file
2. Click on Tranform Data
3. Find the query that is called "FilePath"
4. Update this to the root directory that all of your code is stored in
5. Press "Close & Apply". All of your data should now be correct and up to date

## Things I am wanting to accomplish
1. See what overall makes a winning team, a winning team. Compare stats across years, using a system of rankings